[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$mcpRoot = Split-Path -Parent $PSScriptRoot
$profileFile = Join-Path $mcpRoot "tunnel\nematics3d-local.yaml"
$installDirectory = Join-Path $env:LOCALAPPDATA "Nematics3D\mcp-tunnel"
$tunnelClient = Join-Path $installDirectory "tunnel-client.exe"
$secretFile = Join-Path $installDirectory "runtime-key.dpapi"

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
            throw "tunnel-client doctor failed with exit code $LASTEXITCODE."
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
            $tunnelProcess = Start-Process `
                -FilePath $tunnelClient `
                -ArgumentList @("run", "--profile-file", $profileFile) `
                -NoNewWindow `
                -PassThru

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

                Start-Sleep -Milliseconds 100
                $tunnelProcess.Refresh()
            }
        } finally {
            [Console]::TreatControlCAsInput = $originalTreatControlCAsInput
            if ($null -ne $tunnelProcess) {
                $tunnelProcess.Dispose()
            }
        }

        Write-Host ""
        Write-Host "Nematics3D MCP Tunnel stopped."
        $choice = Read-Host "Enter R to restart or Q to quit"
        if ($choice.Trim().ToUpperInvariant() -ne "R") {
            break
        }
        Write-Host ""
    }
} finally {
    Remove-Item Env:CONTROL_PLANE_API_KEY -ErrorAction SilentlyContinue
}
