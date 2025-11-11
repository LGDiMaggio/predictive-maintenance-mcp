# 🚀 Quick Setup for Claude Desktop Testing

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "MCP Server Setup for Claude Desktop" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 1. Check if Claude config directory exists
$claudeConfigDir = "$env:APPDATA\Claude"
if (-not (Test-Path $claudeConfigDir)) {
    Write-Host "⚠️  Claude config directory not found at: $claudeConfigDir" -ForegroundColor Yellow
    Write-Host "   Creating directory..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $claudeConfigDir -Force | Out-Null
}

$claudeConfigFile = "$claudeConfigDir\claude_desktop_config.json"

# 2. Check if config file exists
if (Test-Path $claudeConfigFile) {
    Write-Host "✓ Claude config file exists" -ForegroundColor Green
    Write-Host "  Location: $claudeConfigFile" -ForegroundColor Gray
    
    # Backup existing config
    $backupFile = "$claudeConfigDir\claude_desktop_config.backup_$(Get-Date -Format 'yyyyMMdd_HHmmss').json"
    Copy-Item $claudeConfigFile $backupFile
    Write-Host "  Backup created: $backupFile" -ForegroundColor Gray
    Write-Host ""
    
    # Read existing config
    $existingConfig = Get-Content $claudeConfigFile -Raw | ConvertFrom-Json
    
    # Check if our server is already configured
    if ($existingConfig.mcpServers.PSObject.Properties.Name -contains "predictive-maintenance") {
        Write-Host "✓ predictive-maintenance server already configured!" -ForegroundColor Green
        Write-Host "  No changes needed." -ForegroundColor Gray
    } else {
        Write-Host "⚠️  Adding predictive-maintenance server to existing config..." -ForegroundColor Yellow
        
        # Add our server
        $existingConfig.mcpServers | Add-Member -MemberType NoteProperty -Name "predictive-maintenance" -Value @{
            command = "uv"
            args = @(
                "--directory",
                "C:\\path\\to\\predictive-maintenance-mcp",
                "run",
                "predictive-maintenance-mcp"
            )
        }
        
        # Save updated config
        $existingConfig | ConvertTo-Json -Depth 10 | Set-Content $claudeConfigFile
        Write-Host "✓ Configuration updated!" -ForegroundColor Green
    }
} else {
    Write-Host "⚠️  Claude config file not found" -ForegroundColor Yellow
    Write-Host "  Creating new config..." -ForegroundColor Yellow
    
    # Create new config from example
    $newConfig = @{
        mcpServers = @{
            "predictive-maintenance" = @{
                command = "uv"
                args = @(
                    "--directory",
                    "C:\\path\\to\\predictive-maintenance-mcp",
                    "run",
                    "predictive-maintenance-mcp"
                )
            }
        }
    }
    
    $newConfig | ConvertTo-Json -Depth 10 | Set-Content $claudeConfigFile
    Write-Host "✓ Configuration created!" -ForegroundColor Green
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Configuration Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "1. Close Claude Desktop completely (if running)" -ForegroundColor White
Write-Host "2. Restart Claude Desktop" -ForegroundColor White
Write-Host "3. Look for MCP icon/indicator" -ForegroundColor White
Write-Host "4. Try: 'List all available signals'" -ForegroundColor White
Write-Host ""
Write-Host "📚 Full testing guide: TEST_WITH_CLAUDE.md" -ForegroundColor Cyan
Write-Host ""

# 3. Verify server can start (quick test)
Write-Host "Testing server startup..." -ForegroundColor Cyan
$projectDir = Split-Path $PSScriptRoot -Parent
if (Test-Path "$projectDir\src\machinery_diagnostics_server.py") {
    try {
        $testResult = python -m py_compile "$projectDir\src\machinery_diagnostics_server.py" 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ Server file syntax OK" -ForegroundColor Green
        } else {
            Write-Host "✗ Server file has syntax errors" -ForegroundColor Red
            Write-Host "  $testResult" -ForegroundColor Red
        }
    } catch {
        Write-Host "⚠️  Could not test server file" -ForegroundColor Yellow
    }
}

# 4. Check data directory
$dataDir = "$projectDir\data\signals"
if (Test-Path $dataDir) {
    $csvCount = (Get-ChildItem -Path $dataDir -Recurse -Filter "*.csv" | Measure-Object).Count
    Write-Host "✓ Data directory exists with $csvCount CSV files" -ForegroundColor Green
} else {
    Write-Host "✗ Data directory not found: $dataDir" -ForegroundColor Red
}

Write-Host ""
Write-Host "Setup complete! 🎉" -ForegroundColor Green
Write-Host ""

