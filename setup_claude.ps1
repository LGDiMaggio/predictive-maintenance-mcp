# Setup for Claude Desktop — installs venv and configures claude_desktop_config.json
#
# Handles two scenarios automatically:
#   - Repo NOT on cloud sync: venv created in <repo>\.venv (standard)
#   - Repo on OneDrive/Dropbox: venv created in %LOCALAPPDATA%\venvs\predictive-maintenance-mcp
#     (avoids >60 s cold-start caused by cloud-sync file scanning)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "MCP Server Setup for Claude Desktop" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$repoPath = $PSScriptRoot

# -----------------------------------------------------------------------
# 1. Locate uv — Claude Desktop needs the FULL path because it launches
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
    Write-Host "✗ uv not found." -ForegroundColor Red
    Write-Host "  Install uv from: https://docs.astral.sh/uv/getting-started/installation/" -ForegroundColor Yellow
    Write-Host "  Then re-run this script." -ForegroundColor Yellow
    exit 1
}
Write-Host "✓ Found uv: $uvExe" -ForegroundColor Green

# -----------------------------------------------------------------------
# 2. Decide venv location — keep it off cloud storage to avoid slow startup
# -----------------------------------------------------------------------
$isCloudSynced = $repoPath -match "OneDrive|Dropbox|Google Drive|SkyDrive|iCloudDrive|Box Sync"
if ($isCloudSynced) {
    $venvPath = "$env:LOCALAPPDATA\venvs\predictive-maintenance-mcp"
    Write-Host ""
    Write-Host "⚠️  Repo is inside a cloud-synced folder." -ForegroundColor Yellow
    Write-Host "   Venv will be stored locally (avoids Claude Desktop timeout):" -ForegroundColor Yellow
    Write-Host "   $venvPath" -ForegroundColor Gray
} else {
    $venvPath = "$repoPath\.venv"
    Write-Host "   Venv location: $venvPath" -ForegroundColor Gray
}

# -----------------------------------------------------------------------
# 3. Create venv + install package (non-editable for local file locality)
# -----------------------------------------------------------------------
Write-Host ""
Write-Host "Creating Python environment..." -ForegroundColor Cyan

New-Item -ItemType Directory -Force (Split-Path $venvPath) | Out-Null
& $uvExe venv $venvPath --python 3.11 --quiet 2>&1 | Out-Null

Write-Host "Installing dependencies (this may take a minute on first run)..." -ForegroundColor Cyan
& $uvExe pip install --python "$venvPath\Scripts\python.exe" $repoPath --quiet 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Dependency installation failed." -ForegroundColor Red
    exit 1
}
Write-Host "✓ Dependencies installed" -ForegroundColor Green

# Pre-compile .pyc files — cuts cold-start time significantly
Write-Host "Pre-compiling bytecode for faster startup..." -ForegroundColor Cyan
& "$venvPath\Scripts\python.exe" -m compileall -q "$venvPath\Lib\site-packages" 2>&1 | Out-Null
Write-Host "✓ Bytecode compiled" -ForegroundColor Green

# -----------------------------------------------------------------------
# 4. Quick smoke-test
# -----------------------------------------------------------------------
Write-Host "Testing server import..." -ForegroundColor Cyan
$importTest = & "$venvPath\Scripts\python.exe" -c "import predictive_maintenance_mcp; print('OK')" 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Server import OK" -ForegroundColor Green
} else {
    Write-Host "✗ Server import failed: $importTest" -ForegroundColor Red
    exit 1
}

# -----------------------------------------------------------------------
# 5. Build the server config entry for claude_desktop_config.json
# -----------------------------------------------------------------------
$pythonExePath = "$venvPath\Scripts\python.exe".Replace('\', '/')
$serverEntry = [ordered]@{
    command = $pythonExePath
    args    = @("-m", "predictive_maintenance_mcp")
}
# If venv is outside the repo, tell the server where to find data/ and models/
if ($isCloudSynced) {
    $serverEntry.env = @{ PDM_PROJECT_DIR = $repoPath.Replace('\', '/') }
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
    Write-Host "✓ Existing config backed up: $backupFile" -ForegroundColor Gray

    $existingConfig = Get-Content $claudeConfigFile -Raw | ConvertFrom-Json

    if (-not $existingConfig.PSObject.Properties["mcpServers"]) {
        $existingConfig | Add-Member -MemberType NoteProperty -Name "mcpServers" -Value ([PSCustomObject]@{})
    }

    # Always update (re-running setup may change venv path)
    if ($existingConfig.mcpServers.PSObject.Properties.Name -contains "predictive-maintenance") {
        $existingConfig.mcpServers.PSObject.Properties.Remove("predictive-maintenance")
        Write-Host "⚠️  Updating existing predictive-maintenance entry..." -ForegroundColor Yellow
    } else {
        Write-Host "⚠️  Adding predictive-maintenance entry..." -ForegroundColor Yellow
    }
    $existingConfig.mcpServers | Add-Member -MemberType NoteProperty -Name "predictive-maintenance" -Value $serverEntry
    $existingConfig | ConvertTo-Json -Depth 10 | Set-Content $claudeConfigFile -Encoding UTF8
    Write-Host "✓ Configuration updated!" -ForegroundColor Green

} else {
    Write-Host ""
    Write-Host "⚠️  No existing Claude config found — creating new one." -ForegroundColor Yellow
    $newConfig = [ordered]@{ mcpServers = [ordered]@{ "predictive-maintenance" = $serverEntry } }
    $newConfig | ConvertTo-Json -Depth 10 | Set-Content $claudeConfigFile -Encoding UTF8
    Write-Host "✓ Configuration created at: $claudeConfigFile" -ForegroundColor Green
}

# -----------------------------------------------------------------------
# 7. Check data directory
# -----------------------------------------------------------------------
$dataDir = "$repoPath\data\signals"
if (Test-Path $dataDir) {
    $csvCount = (Get-ChildItem -Path $dataDir -Recurse -Filter "*.csv" -ErrorAction SilentlyContinue | Measure-Object).Count
    Write-Host "✓ Data directory OK ($csvCount CSV files)" -ForegroundColor Green
} else {
    Write-Host "⚠️  data\signals\ not found — server will start but no sample signals available." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Close Claude Desktop completely (File > Quit, not just minimize)" -ForegroundColor White
Write-Host "  2. Restart Claude Desktop" -ForegroundColor White
Write-Host "  3. Look for the hammer/tools icon — predictive-maintenance should be listed" -ForegroundColor White
Write-Host "  4. Try: 'List all available signals'" -ForegroundColor White
Write-Host ""
Write-Host "  Python: $pythonExePath" -ForegroundColor Gray
Write-Host "  Config: $claudeConfigFile" -ForegroundColor Gray
Write-Host ""

