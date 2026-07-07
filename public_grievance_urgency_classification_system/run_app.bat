@echo off
REM Start both platforms (citizen portal + Model Observatory) on http://localhost:8017
cd /d "%~dp0"
if exist "..\.venv\Scripts\python.exe" (
  set PY=..\.venv\Scripts\python.exe
) else if exist ".venv\Scripts\python.exe" (
  set PY=.venv\Scripts\python.exe
) else (
  set PY=python
)
echo Citizen portal:    http://localhost:8017
echo Model Observatory: http://localhost:8017/admin
%PY% -m uvicorn app.server:app --port 8017
