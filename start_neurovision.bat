@echo off
title NeuroVision - Brain Tumor Detection

cd /d "%~dp0"

echo ==========================================
echo       NEUROVISION - BRAIN TUMOR AI
echo ==========================================
echo.
echo Starting NeuroVision...
echo.

start "" http://127.0.0.1:5000

"D:\Program Files\Python313\python.exe" app.py

pause