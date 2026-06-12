Set WShell = CreateObject("WScript.Shell")
WShell.Run "powershell.exe -ExecutionPolicy Bypass -NonInteractive -WindowStyle Hidden -File ""C:\Users\ninad\Edgeful\orb_ib_backtester\edgeful_launcher.ps1""", 0, False
