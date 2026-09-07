[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$mcpRoot = Split-Path -Parent $PSScriptRoot
$profileFile = Join-Path $mcpRoot "tunnel\nematics3d-local.yaml"
$installDirectory = Join-Path $env:LOCALAPPDATA "Nematics3D\mcp-tunnel"
$tunnelClient = Join-Path $installDirectory "tunnel-client.exe"
$secretFile = Join-Path $installDirectory "runtime-key.dpapi"
$healthUrlFile = Join-Path $installDirectory "health-url.txt"
$restartDelaySeconds = 30
$healthCheckIntervalSeconds = 10
$healthStartupGraceSeconds = 45
$stalePollThresholdSeconds = 90
$healthFailureLimit = 3

function Test-TunnelControlPlaneHealth {
    param(
        [System.Diagnostics.Process]$Process,
        [string]$UrlFile,
        [int]$StaleThresholdSeconds
    )

    if (-not (Test-Path -LiteralPath $UrlFile -PathType Leaf)) {
        return $false
    }

    $healthOutput = & $tunnelClient health `
        --url-file $UrlFile `
        --pid $Process.Id `
        --require-control-plane-poll `
        --json 2>$null | Out-String
    if ($LASTEXITCODE -ne 0) {
        return $false
    }

    try {
        $health = $healthOutput | ConvertFrom-Json
        $lastPollSeconds = [long]$health.control_plane_poll.value
        $lastPoll = [DateTimeOffset]::FromUnixTimeSeconds($lastPollSeconds)
        $pollAge = [DateTimeOffset]::UtcNow - $lastPoll
        return (
            $health.result -eq "ok" -and
            $health.process.running -eq $true -and
            $pollAge.TotalSeconds -le $StaleThresholdSeconds
        )
    } catch {
        return $false
    }
}

function Wait-ForTunnelRestart {
    param(
        [int]$DelaySeconds
    )

    Write-Host ""
    Write-Host (
        "Retrying in {0} seconds. Press R to retry now or C to stop." -f `
            $DelaySeconds
    )
    $deadline = [DateTime]::UtcNow.AddSeconds($DelaySeconds)

    while ([DateTime]::UtcNow -lt $deadline) {
        if ([Console]::KeyAvailable) {
            $key = [Console]::ReadKey($true)
            if ($key.Key -eq [ConsoleKey]::R) {
                Write-Host "Retry requested."
                return $true
            }
            if ($key.Key -eq [ConsoleKey]::C) {
                Write-Host "Reconnect loop stopped."
                return $false
            }
        }

        Start-Sleep -Milliseconds 100
    }

    Write-Host "Retry timer elapsed."
    return $true
}

if (-not (Test-Path -LiteralPath $profileFile -PathType Leaf)) {
    throw "Tunnel profile not found: $profileFile"
}

if (-not (Test-Path -LiteralPath $tunnelClient -PathType Leaf)) {
    throw "tunnel-client is not installed. Run setup-one-click.ps1 first."
}

if (-not (Test-Path -LiteralPath $secretFile -PathType Leaf)) {
    throw "Runtime key is not configured. Run setup-one-click.ps1 first."
}

$encryptedKey = (Get-Content -Raw -LiteralPath $secretFile).Trim()
$secureKey = ConvertTo-SecureString $encryptedKey
$credential = [System.Net.NetworkCredential]::new("", $secureKey)
$env:CONTROL_PLANE_API_KEY = $credential.Password

try {
    while ($true) {
        Write-Host "Checking Nematics3D MCP Tunnel..."
        & $tunnelClient doctor --profile-file $profileFile
        if ($LASTEXITCODE -ne 0) {
            Write-Warning (
                "tunnel-client doctor failed with exit code {0}." -f `
                    $LASTEXITCODE
            )
            if (-not (Wait-ForTunnelRestart $restartDelaySeconds)) {
                break
            }
            continue
        }

        Write-Host ""
        Write-Host "Starting Nematics3D MCP Tunnel."
        Write-Host "Keep this window open; press Ctrl+C to stop."
        Write-Host ""

        $originalTreatControlCAsInput = [Console]::TreatControlCAsInput
        $tunnelProcess = $null
        try {
            # Keep Ctrl+C from terminating this wrapper. The wrapper stops only
            # the tunnel process, then continues to the restart prompt below.
            [Console]::TreatControlCAsInput = $true
            Remove-Item -LiteralPath $healthUrlFile -Force -ErrorAction SilentlyContinue
            $tunnelProcess = Start-Process `
                -FilePath $tunnelClient `
                -ArgumentList @(
                    "run",
                    "--profile-file",
                    $profileFile,
                    "--health.url-file",
                    $healthUrlFile
                ) `
                -NoNewWindow `
                -PassThru

            $healthFailureCount = 0
            $nextHealthCheck = [DateTime]::UtcNow.AddSeconds(
                $healthStartupGraceSeconds
            )
            while (-not $tunnelProcess.HasExited) {
                if ([Console]::KeyAvailable) {
                    $key = [Console]::ReadKey($true)
                    $isControlC = (
                        $key.Key -eq [ConsoleKey]::C -and
                        ($key.Modifiers -band [ConsoleModifiers]::Control)
                    )
                    if ($isControlC) {
                        Write-Host ""
                        Write-Host "Stopping Nematics3D MCP Tunnel..."
                        & taskkill.exe /PID $tunnelProcess.Id /T /F 2>&1 | Out-Null
                        $tunnelProcess.WaitForExit()
                        break
                    }
                }

                if ([DateTime]::UtcNow -ge $nextHealthCheck) {
                    $isHealthy = Test-TunnelControlPlaneHealth `
                        -Process $tunnelProcess `
                        -UrlFile $healthUrlFile `
                        -StaleThresholdSeconds $stalePollThresholdSeconds
                    if ($isHealthy) {
                        $healthFailureCount = 0
                    } else {
                        $healthFailureCount += 1
                        Write-Warning (
                            "Tunnel health check failed ({0}/{1})." -f `
                                $healthFailureCount,
                                $healthFailureLimit
                        )
                    }

                    if ($healthFailureCount -ge $healthFailureLimit) {
                        Write-Warning (
                            "Tunnel is alive but its control-plane connection " +
                            "is unhealthy. Restarting it."
                        )
                        & taskkill.exe /PID $tunnelProcess.Id /T /F 2>&1 |
                            Out-Null
                        $tunnelProcess.WaitForExit()
                        break
                    }

                    $nextHealthCheck = [DateTime]::UtcNow.AddSeconds(
                        $healthCheckIntervalSeconds
                    )
                }

                Start-Sleep -Milliseconds 100
                $tunnelProcess.Refresh()
            }
        } finally {
            [Console]::TreatControlCAsInput = $originalTreatControlCAsInput
            if ($null -ne $tunnelProcess) {
                if (-not $tunnelProcess.HasExited) {
                    & taskkill.exe /PID $tunnelProcess.Id /T /F 2>&1 | Out-Null
                    $tunnelProcess.WaitForExit()
                }
                $tunnelProcess.Dispose()
            }
        }

        Write-Host ""
        Write-Host "Nematics3D MCP Tunnel stopped."
        if (-not (Wait-ForTunnelRestart $restartDelaySeconds)) {
            break
        }
        Write-Host ""
    }
} finally {
    Remove-Item -LiteralPath $healthUrlFile -Force -ErrorAction SilentlyContinue
    Remove-Item Env:CONTROL_PLANE_API_KEY -ErrorAction SilentlyContinue
}
