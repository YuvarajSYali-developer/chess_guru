@echo off
setlocal

set ROOT=%~dp0
cd /d "%ROOT%"
set PYTHONUTF8=1

echo Starting backend on http://127.0.0.1:8000 ...
start "Chess Backend" cmd /k "cd /d "%ROOT%" && python -m uvicorn backend.app:app --reload --port 8000 --app-dir "%ROOT%""

echo Starting frontend on http://127.0.0.1:8080 ...
start "Chess Frontend" cmd /k "cd /d "%ROOT%frontend" && python -m http.server 8080"

echo Waiting for backend to be ready...
for /l %%i in (1,1,30) do (
  powershell -NoProfile -Command "try { (Invoke-WebRequest http://127.0.0.1:8000/api/health -UseBasicParsing -TimeoutSec 2) | Out-Null; exit 0 } catch { exit 1 }"
  if not errorlevel 1 goto OPEN_BROWSER
  timeout /t 1 > nul
)

:OPEN_BROWSER
start http://127.0.0.1:8080/?v=1
echo Done. Keep the backend and frontend windows open.
