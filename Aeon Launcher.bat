@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" "scripts\aeon_launcher.py"
) else (
  python "scripts\aeon_launcher.py"
)
