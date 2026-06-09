@echo off
REM Detached launcher for the Probability Explorer (port 8502)
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
"C:\Users\ninad\AppData\Local\Python\pythoncore-3.14-64\python.exe" -m streamlit run prob_app.py --server.headless=true --server.port=8502 --browser.gatherUsageStats=false > "%~dp0logs\prob.log" 2>&1
