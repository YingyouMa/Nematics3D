@echo off
set "SCRIPT_DIR=%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%scripts\start-nematics3d-mcp.ps1"
if errorlevel 1 (
    echo.
    echo Startup failed. Review the message above.
    pause
)
