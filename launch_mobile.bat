@echo off
REM ================================================================
REM  TradeIQ Mobile Launcher
REM  Run this .bat file to start the dashboard AND expose it via ngrok
REM  Then open the ngrok URL on your phone
REM ================================================================

title TradeIQ Mobile Launcher

echo.
echo  =============================================
echo   TradeIQ - Mobile Access Launcher
echo  =============================================
echo.

REM Set UTF-8 mode for emoji support
set PYTHONUTF8=1

REM Step 1: Activate virtual environment if it exists
REM (uncomment the next line if you use a venv)
REM call .venv\Scripts\activate.bat

REM Step 2: Check if ngrok is installed
where ngrok >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [!] ngrok not found. Downloading now...
    echo.
    echo Please download ngrok from https://ngrok.com/download
    echo and add it to your PATH (or place ngrok.exe in this folder).
    echo.
    echo ALTERNATIVE - Use the Wi-Fi URL instead:
    echo Open a command prompt and run:
    echo   ipconfig
    echo Then open http://YOUR-PC-IP:8501 on your phone.
    echo.
    pause
    goto :WIFI_ONLY
)

REM Step 3: Start Streamlit in background
echo [1/3] Starting Streamlit dashboard...
start "TradeIQ-Streamlit" cmd /k "cd /d %~dp0 && set PYTHONUTF8=1 && python -m streamlit run app.py --server.port 8501 --server.headless true"
timeout /t 4 /nobreak >nul

REM Step 4: Start ngrok tunnel
echo [2/3] Opening ngrok tunnel...
start "TradeIQ-ngrok" cmd /k "ngrok http 8501"
timeout /t 3 /nobreak >nul

REM Step 5: Print instructions
echo.
echo [3/3] Done!
echo.
echo ============================================================
echo  HOW TO ACCESS ON YOUR MOBILE:
echo ============================================================
echo  1. Look at the ngrok window that just opened
echo  2. Copy the "Forwarding" URL (e.g. https://abc123.ngrok.io)
echo  3. Open that URL on your phone browser
echo  4. On iPhone: Tap Share > "Add to Home Screen"
echo  5. On Android: Tap menu > "Add to Home Screen"
echo ============================================================
echo.
echo  LOCAL (same Wi-Fi only): http://localhost:8501
echo ============================================================
echo.
pause
goto :EOF

:WIFI_ONLY
echo [1/1] Starting Streamlit (Wi-Fi access only)...
echo.
echo Finding your PC's IP address:
ipconfig | findstr /i "IPv4"
echo.
echo Open the address shown above (port 8501) on your phone.
echo Example: http://192.168.1.100:8501
echo.
start "TradeIQ-Streamlit" cmd /k "cd /d %~dp0 && set PYTHONUTF8=1 && python -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0"
pause
