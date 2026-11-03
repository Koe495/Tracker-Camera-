@echo off
cd /d "%~dp0"
".venv-kinect\Scripts\python.exe" main.py
if errorlevel 1 pause
