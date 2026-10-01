@echo off
title n8n Cloud Webhook Bridge
echo =========================================================
echo Start Python Webhook Server + ngrok for n8n Cloud
echo =========================================================
echo.

REM 1. Check for ngrok.exe
if not exist "%~dp0\ngrok.exe" (
    echo [!] ngrok.exe not found! Please run 'python install_ngrok.py' first.
    pause
    exit /b
)

REM 2. Start Webhook Server in a new window
echo Starting Webhook Server on port 8502...
start "TradeIQ-Webhook" cmd /k "cd /d %~dp0 && python webhook_server.py"

REM 3. Wait a moment
timeout /t 4 /nobreak >nul

REM 4. Start ngrok in this window
echo Starting ngrok...
echo.
echo =======================================================
echo CRITICAL NEXT STEP FOR N8N CLOUD:
echo Look directly above for the "Forwarding" URL.
echo It looks like: https://abc-123.ngrok-free.app
echo 
echo In your n8n Cloud workflow, replace "http://localhost:8502"
echo with that Forwarding URL. For example:
echo https://abc-123.ngrok-free.app/webhook/sentiment
echo =======================================================
echo.
"%~dp0\ngrok.exe" http 8502
