@echo off
title n8n Local Launcher (TradeIQ)
echo =======================================================
echo Starting n8n with Local Network Access Enabled
echo =======================================================
echo This allows n8n's HTTP Request nodes to connect to our 
echo FastAPI webhook server (http://localhost:8502) for:
echo   - VADER Sentiment + DB Save
echo   - ML Predictions
echo   - Auto-Trading Cycles
echo =======================================================

REM Disable SSRF protection so n8n can access localhost/127.0.0.1
set N8N_BLOCK_OUTGOING_IPS_BY_DEFAULT=false

REM Start n8n using npx
echo Starting n8n...
npx n8n
