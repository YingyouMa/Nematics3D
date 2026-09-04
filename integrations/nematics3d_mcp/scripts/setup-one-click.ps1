[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$releaseDirectory = Join-Path `
    $env:USERPROFILE `
    "Downloads\openai-tunnel-client-v0.0.14"
$installDirectory = Join-Path $env:LOCALAPPDATA "Nematics3D\mcp-tunnel"
$secretFile = Join-Path $installDirectory "runtime-key.dpapi"
$requiredFiles = @(
    "tunnel-client.exe",
    "cloudflared.exe",
    "cloudflared-manifest.json"
)

foreach ($fileName in $requiredFiles) {
    $sourcePath = Join-Path $releaseDirectory $fileName
    if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
        throw "Required tunnel-client file not found: $sourcePath"
    }
}

New-Item -ItemType Directory -Path $installDirectory -Force | Out-Null

foreach ($fileName in $requiredFiles) {
    Copy-Item `
        -LiteralPath (Join-Path $releaseDirectory $fileName) `
        -Destination (Join-Path $installDirectory $fileName) `
        -Force
}

if ($env:CONTROL_PLANE_API_KEY) {
    $runtimeKey = ConvertTo-SecureString `
        $env:CONTROL_PLANE_API_KEY `
        -AsPlainText `
        -Force
    Write-Host "Using CONTROL_PLANE_API_KEY from this PowerShell session."
} else {
    $runtimeKey = Read-Host `
        "Paste the OpenAI Tunnel Runtime API key" `
        -AsSecureString
}

$encryptedKey = ConvertFrom-SecureString $runtimeKey
Set-Content -LiteralPath $secretFile -Value $encryptedKey -Encoding utf8

Write-Host ""
Write-Host "One-click setup complete."
Write-Host "The Runtime API key is protected with Windows DPAPI for this user."
Write-Host "Start the tunnel with start-nematics3d-mcp.cmd."
