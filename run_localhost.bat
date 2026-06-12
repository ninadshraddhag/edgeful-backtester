@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================
echo    Edgeful Strategy Suite  -  localhost:8501
echo ============================================
echo.

REM ---------------------------------------------------------------
REM  If the app is already running on 8501 (e.g. via the
REM  start_edgeful.vbs background launcher), don't start a second
REM  instance -- just open the browser to it.
REM ---------------------------------------------------------------
netstat -ano | findstr ":8501" | findstr "LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo App is already running -- opening http://localhost:8501
    start "" "http://localhost:8501"
    timeout /t 3 >nul
    exit /b 0
)

REM ---------------------------------------------------------------
REM  Find a REAL Python interpreter.
REM  Candidate 1: the verified install used by the background
REM  launcher (has all dependencies). Then py launcher, then PATH
REM  python (guarding against the Microsoft Store stub).
REM ---------------------------------------------------------------
set "PY_CMD="
set "KNOWN_PY=C:\Users\ninad\AppData\Local\Python\pythoncore-3.14-64\python.exe"

if exist "%KNOWN_PY%" set "PY_CMD="%KNOWN_PY%""

if not defined PY_CMD (
    for /f "delims=" %%v in ('py -3 --version 2^>^&1') do set "VER=%%v"
    echo !VER! | findstr /b /c:"Python 3" >nul && set "PY_CMD=py -3"
)

if not defined PY_CMD (
    for /f "delims=" %%v in ('python --version 2^>^&1') do set "VER=%%v"
    echo !VER! | findstr /b /c:"Python 3" >nul && set "PY_CMD=python"
)

if not defined PY_CMD (
    echo [ERROR] No working Python found. Install Python 3.11+ from python.org
    echo         and tick "Add python.exe to PATH" during setup.
    pause
    exit /b 1
)

REM ---------------------------------------------------------------
REM  Install dependencies only if missing (first run).
REM ---------------------------------------------------------------
%PY_CMD% -c "import streamlit, pandas, numpy, plotly" >nul 2>&1
if errorlevel 1 (
    echo Installing required packages ^(first run only^)...
    %PY_CMD% -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Dependency install failed -- see messages above.
        pause
        exit /b 1
    )
)

REM ---------------------------------------------------------------
REM  Launch on localhost:8501. Browser opens automatically.
REM  Keep this window open; Ctrl+C stops the app.
REM ---------------------------------------------------------------
echo Starting Streamlit on http://localhost:8501 ...
echo.
%PY_CMD% -m streamlit run app.py --server.port 8501 --server.address localhost

echo.
pause
exit /b 0
