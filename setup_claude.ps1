# Setup for Claude Desktop - installs venv and configures claude_desktop_config.json
#
# Handles two scenarios automatically:
#   - Repo NOT on cloud sync: venv created in <repo>\.venv (standard)
#   - Repo on OneDrive/Dropbox: venv created in %LOCALAPPDATA%\venvs\predictive-maintenance-mcp
#     (avoids >60 s cold-start caused by cloud-sync file scanning)
#
# Runs under Windows PowerShell 5.1 and PowerShell 7. Keep this file ASCII-only:
# 5.1 reads a BOM-less file as ANSI and breaks on non-ASCII characters.

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "MCP Server Setup for Claude Desktop" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$repoPath = $PSScriptRoot

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

# Run a native command and return its exit code and merged output.
# Under Windows PowerShell 5.1 a line on stderr becomes a terminating
# NativeCommandError while $ErrorActionPreference is "Stop", so the
# preference is relaxed around the call and the exit code decides.
function Invoke-Native {
    param(
        [Parameter(Mandatory = $true)][string]$Exe,
        [string[]]$Arguments = @()
    )
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $lines = & $Exe @Arguments 2>&1 | ForEach-Object { "$_" }
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previous
    }
    return @{ Code = $code; Output = (@($lines) -join "`n") }
}

# Write a JSON file as UTF-8 without BOM in every PowerShell edition:
# Set-Content -Encoding UTF8 adds a BOM under 5.1, which JSON parsers reject.
function Write-JsonFile {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)]$Object
    )
    $json = $Object | ConvertTo-Json -Depth 10
    $encoding = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($Path, $json, $encoding)
}

# -----------------------------------------------------------------------
# 1. Locate uv - Claude Desktop needs the FULL path because it launches
#    processes with a minimal PATH that often omits user-local tool dirs.
# -----------------------------------------------------------------------
function Find-Uv {
    # Try PATH first (works on machines where uv is in a system-level location)
    $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
    if ($uvCmd) { return $uvCmd.Source }
    # Common user-install locations (uv installer default on Windows)
    $candidates = @(
        "$env:USERPROFILE\.local\bin\uv.exe",
        "$env:APPDATA\uv\bin\uv.exe",
        "$env:LOCALAPPDATA\uv\bin\uv.exe"
    )
    foreach ($p in $candidates) { if (Test-Path $p) { return $p } }
    return $null
}

$uvExe = Find-Uv
if (-not $uvExe) {
    Write-Host "[X] uv not found." -ForegroundColor Red
    Write-Host "    Install uv from: https://docs.astral.sh/uv/getting-started/installation/" -ForegroundColor Yellow
    Write-Host "    Then re-run this script." -ForegroundColor Yellow
    exit 1
}
Write-Host "[OK] Found uv: $uvExe" -ForegroundColor Green

# -----------------------------------------------------------------------
# 2. Decide venv location - keep it off cloud storage to avoid slow startup
# -----------------------------------------------------------------------
$isCloudSynced = $repoPath -match "OneDrive|Dropbox|Google Drive|SkyDrive|iCloudDrive|Box Sync"
if ($isCloudSynced) {
    $venvPath = "$env:LOCALAPPDATA\venvs\predictive-maintenance-mcp"
    Write-Host ""
    Write-Host "[!] Repo is inside a cloud-synced folder." -ForegroundColor Yellow
    Write-Host "    Venv will be stored locally (avoids Claude Desktop timeout):" -ForegroundColor Yellow
    Write-Host "    $venvPath" -ForegroundColor Gray
} else {
    $venvPath = "$repoPath\.venv"
    Write-Host "    Venv location: $venvPath" -ForegroundColor Gray
}

# -----------------------------------------------------------------------
# 3. Create venv + install package (non-editable for local file locality)
# -----------------------------------------------------------------------
Write-Host ""
Write-Host "Creating Python environment..." -ForegroundColor Cyan

New-Item -ItemType Directory -Force (Split-Path $venvPath) | Out-Null
$venvResult = Invoke-Native -Exe $uvExe -Arguments @("venv", $venvPath, "--python", "3.11", "--quiet")
if ($venvResult.Code -ne 0) {
    Write-Host "[X] Could not create the virtual environment." -ForegroundColor Red
    Write-Host $venvResult.Output -ForegroundColor Gray
    exit 1
}

Write-Host "Installing dependencies (this may take a minute on first run)..." -ForegroundColor Cyan
$installResult = Invoke-Native -Exe $uvExe -Arguments @("pip", "install", "--python", "$venvPath\Scripts\python.exe", $repoPath, "--quiet")
if ($installResult.Code -ne 0) {
    Write-Host "[X] Dependency installation failed." -ForegroundColor Red
    Write-Host $installResult.Output -ForegroundColor Gray
    exit 1
}
Write-Host "[OK] Dependencies installed" -ForegroundColor Green

# Pre-compile .pyc files - cuts cold-start time significantly
Write-Host "Pre-compiling bytecode for faster startup..." -ForegroundColor Cyan
$null = Invoke-Native -Exe "$venvPath\Scripts\python.exe" -Arguments @("-m", "compileall", "-q", "$venvPath\Lib\site-packages")
Write-Host "[OK] Bytecode compiled" -ForegroundColor Green

# -----------------------------------------------------------------------
# 4. Quick smoke-test
# -----------------------------------------------------------------------
Write-Host "Testing server import..." -ForegroundColor Cyan
$importResult = Invoke-Native -Exe "$venvPath\Scripts\python.exe" -Arguments @("-c", "import predictive_maintenance_mcp; print('OK')")
if ($importResult.Code -eq 0) {
    Write-Host "[OK] Server import OK" -ForegroundColor Green
} else {
    Write-Host "[X] Server import failed: $($importResult.Output)" -ForegroundColor Red
    exit 1
}

# -----------------------------------------------------------------------
# 5. Build the server config entry for claude_desktop_config.json
# -----------------------------------------------------------------------
$pythonExePath = "$venvPath\Scripts\python.exe".Replace('\', '/')
$serverEntry = [ordered]@{
    command = $pythonExePath
    args    = @("-m", "predictive_maintenance_mcp")
    # The package is installed non-editable, so the server cannot find the
    # repo's data/ and models/ on its own: tell it where the clone lives,
    # whatever the venv location.
    env     = @{ PDM_PROJECT_DIR = $repoPath.Replace('\', '/') }
}

# -----------------------------------------------------------------------
# 6. Write / merge claude_desktop_config.json
# -----------------------------------------------------------------------
$claudeConfigDir  = "$env:APPDATA\Claude"
$claudeConfigFile = "$claudeConfigDir\claude_desktop_config.json"

New-Item -ItemType Directory -Force $claudeConfigDir | Out-Null

if (Test-Path $claudeConfigFile) {
    $backupFile = "$claudeConfigDir\claude_desktop_config.backup_$(Get-Date -Format 'yyyyMMdd_HHmmss').json"
    Copy-Item $claudeConfigFile $backupFile
    Write-Host ""
    Write-Host "[OK] Existing config backed up: $backupFile" -ForegroundColor Gray

    $existingConfig = Get-Content $claudeConfigFile -Raw | ConvertFrom-Json

    if (-not $existingConfig.PSObject.Properties["mcpServers"]) {
        $existingConfig | Add-Member -MemberType NoteProperty -Name "mcpServers" -Value ([PSCustomObject]@{})
    }

    # Always update (re-running setup may change venv path)
    if ($existingConfig.mcpServers.PSObject.Properties.Name -contains "predictive-maintenance") {
        $existingConfig.mcpServers.PSObject.Properties.Remove("predictive-maintenance")
        Write-Host "[!] Updating existing predictive-maintenance entry..." -ForegroundColor Yellow
    } else {
        Write-Host "[!] Adding predictive-maintenance entry..." -ForegroundColor Yellow
    }
    $existingConfig.mcpServers | Add-Member -MemberType NoteProperty -Name "predictive-maintenance" -Value $serverEntry
    Write-JsonFile -Path $claudeConfigFile -Object $existingConfig
    Write-Host "[OK] Configuration updated!" -ForegroundColor Green

} else {
    Write-Host ""
    Write-Host "[!] No existing Claude config found - creating new one." -ForegroundColor Yellow
    $newConfig = [ordered]@{ mcpServers = [ordered]@{ "predictive-maintenance" = $serverEntry } }
    Write-JsonFile -Path $claudeConfigFile -Object $newConfig
    Write-Host "[OK] Configuration created at: $claudeConfigFile" -ForegroundColor Green
}

# -----------------------------------------------------------------------
# 7. Check data directory
# -----------------------------------------------------------------------
$dataDir = "$repoPath\data\signals"
if (Test-Path $dataDir) {
    $csvCount = (Get-ChildItem -Path $dataDir -Recurse -Filter "*.csv" -ErrorAction SilentlyContinue | Measure-Object).Count
    Write-Host "[OK] Data directory OK ($csvCount CSV files)" -ForegroundColor Green
} else {
    Write-Host "[!] data\signals\ not found - server will start but no sample signals available." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Quit Claude Desktop completely (File > Quit, not just the window)" -ForegroundColor White
Write-Host "  2. Start Claude Desktop again" -ForegroundColor White
Write-Host "  3. In a new chat, ask: 'List the available vibration signals' (the 20 included samples should appear)" -ForegroundColor White
Write-Host "  4. Then try the first prompt from the README Quick Start" -ForegroundColor White
Write-Host ""
Write-Host "  Python: $pythonExePath" -ForegroundColor Gray
Write-Host "  Config: $claudeConfigFile" -ForegroundColor Gray
Write-Host ""
