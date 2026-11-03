@echo off
cd /d "%~dp0"
".venv-kinect\Scripts\python.exe" depth_lab.py
if errorlevel 1 pause
