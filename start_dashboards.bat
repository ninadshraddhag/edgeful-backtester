@echo off
REM ============================================================
REM  One-click launcher for BOTH dashboards.
REM  Double-click this file. Two server windows open, then your
REM  browser opens both apps. Keep the server windows open while
REM  using the apps; close them to stop the servers.
REM ============================================================
cd /d "%~dp0"

echo Starting Edge Backtester (port 8501)...
start "Edge Backtester (8501)" cmd /c "%~dp0serve_app.bat"

echo Starting Probability Explorer (port 8502)...
start "Probability Explorer (8502)" cmd /c "%~dp0serve_prob.bat"

echo Waiting for servers to boot...
timeout /t 13 /nobreak >nul

start "" http://localhost:8501
start "" http://localhost:8502

echo.
echo   Edge Backtester        http://localhost:8501
echo   Probability Explorer   http://localhost:8502
echo.
echo Two server windows are now running. You can close THIS window.
echo To stop the apps, close those two server windows.
timeout /t 6 /nobreak >nul
