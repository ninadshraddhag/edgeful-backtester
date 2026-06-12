$PYTHON  = 'C:\Users\ninad\AppData\Local\Python\pythoncore-3.14-64\python.exe'
$APPDIR  = 'C:\Users\ninad\Edgeful\orb_ib_backtester'
$LOGFILE = 'C:\Users\ninad\Edgeful\orb_ib_backtester\logs\app.log'

while ($true) {
    $proc = Start-Process -FilePath $PYTHON `
        -ArgumentList "-m streamlit run `"$APPDIR\app.py`" --server.port 8501 --server.headless true" `
        -WorkingDirectory $APPDIR `
        -RedirectStandardOutput $LOGFILE `
        -RedirectStandardError  "$APPDIR\logs\app_err.log" `
        -PassThru -NoNewWindow
    $proc.WaitForExit()
    Start-Sleep -Seconds 5   # brief pause before restart on crash
}
