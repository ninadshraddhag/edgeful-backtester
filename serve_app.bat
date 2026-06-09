@echo off
REM Detached launcher for the Edge Backtester (port 8501)
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
"C:\Users\ninad\AppData\Local\Python\pythoncore-3.14-64\python.exe" -m streamlit run app.py --server.headless=true --server.port=8501 --browser.gatherUsageStats=false > "%~dp0logs\app.log" 2>&1
