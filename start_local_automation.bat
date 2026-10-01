@echo off
title TradeIQ Local Automation Starter
echo =======================================================
echo TradeIQ Local Automation Starter
echo =======================================================
echo.
echo This will:
echo   1. Start the FastAPI webhook bridge on port 8502
echo   2. Launch n8n with localhost HTTP access enabled
echo   3. Let you import and run workflows\n8n\n8n_workflow_FINAL_LOCAL.json
echo.

echo Starting webhook server in a new window...
start "TradeIQ-Webhook" cmd /k "cd /d %~dp0 && python webhook_server.py"

echo Waiting for webhook server startup...
timeout /t 4 /nobreak >nul

echo Launching n8n...
call "%~dp0launch_n8n.bat"
