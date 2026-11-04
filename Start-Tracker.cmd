@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run setup.ps1 in PowerShell first.
  pause
  exit /b 1
)
choice /C TU /N /M "Choose [T] Tennis ball or [U] UAV: "
if "%errorlevel%"=="2" goto uav
if not "%errorlevel%"=="1" exit /b 1
set "detector=tennis"
goto run
:uav
set "detector=uav"
:run
".venv\Scripts\python.exe" -B -m rgbd.live camera --detector %detector% %*
set "result=%errorlevel%"
if not "%result%"=="0" pause
exit /b %result%
