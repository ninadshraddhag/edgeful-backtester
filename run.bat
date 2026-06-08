@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ===========================================
echo    ORB ^& IB Strategy Backtester
echo ===========================================
echo.

REM ---------------------------------------------------------------
REM  Find a REAL Python interpreter.
REM  We validate the output starts with "Python 3" so the broken
REM  Microsoft Store stub (prints "Python was not found...") is
REM  rejected instead of being treated as a real interpreter.
REM ---------------------------------------------------------------
set "PY_CMD="

REM --- Candidate 1: the py launcher ---
for /f "delims=" %%v in ('py -3 --version 2^>^&1') do set "VER=%%v"
echo !VER! | findstr /b /c:"Python 3" >nul && set "PY_CMD=py -3"

REM --- Candidate 2: plain "python" (guard against Store stub) ---
if not defined PY_CMD (
    for /f "delims=" %%v in ('python --version 2^>^&1') do set "VER=%%v"
    echo !VER! | findstr /b /c:"Python 3" >nul && set "PY_CMD=python"
)

if not defined PY_CMD goto :no_python

echo Found: !VER!   ^(command: !PY_CMD!^)
echo.

REM ---------------------------------------------------------------
REM  Install dependencies only if they are missing (first run).
REM ---------------------------------------------------------------
echo Checking dependencies...
!PY_CMD! -c "import streamlit, pandas, numpy, plotly" >nul 2>&1
if errorlevel 1 (
    echo Installing required packages ^(first run only, may take a minute^)...
    echo.
    !PY_CMD! -m pip install --upgrade pip
    !PY_CMD! -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo [ERROR] Failed to install dependencies. See the messages above.
        echo.
        pause
        exit /b 1
    )
) else (
    echo All dependencies already installed.
)

echo.
echo Starting Streamlit -- a browser tab will open automatically.
echo Keep this window open while using the app.
echo Press Ctrl+C here to stop.
echo.
!PY_CMD! -m streamlit run app.py

echo.
pause
exit /b 0

REM ---------------------------------------------------------------
:no_python
echo [ERROR] No working Python interpreter was found.
echo.
echo The only "python" on this PC is the Microsoft Store stub,
echo which does not actually run Python.
echo.
echo To fix this:
echo   1. Download Python 3.11 or 3.12 from:
echo          https://www.python.org/downloads/
echo   2. During setup, CHECK the box "Add python.exe to PATH".
echo      ^(Do NOT use the Microsoft Store version.^)
echo   3. Close this window, reopen it, and double-click run.bat again.
echo.
pause
exit /b 1
