@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run setup.ps1 in PowerShell first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -B -m rgbd.live camera %*
if errorlevel 1 pause
