@echo off
set "SCRIPT_DIR=%~dp0"
start "Nematics3D MCP" powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%scripts\start-nematics3d-mcp.ps1"
