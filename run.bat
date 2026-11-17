@echo off
rem Live Clipper launcher: sets up its Python environment on first run, then opens the page.
setlocal
set "VENV=%USERPROFILE%\.live-clipper\venv"
cd /d "%~dp0"

where ffmpeg >nul 2>nul || (echo FFmpeg is not installed. Install it with:  winget install Gyan.FFmpeg & pause & exit /b 1)

if not exist "%VENV%\Scripts\python.exe" (
  echo First run: installing what Live Clipper needs. This takes a few minutes...
  python -m venv "%VENV%" || (echo Python 3.10+ is required: https://www.python.org/downloads/ & pause & exit /b 1)
  "%VENV%\Scripts\python" -m pip install -q --upgrade pip
  "%VENV%\Scripts\python" -m pip install -q -r requirements.txt || (pause & exit /b 1)
)

"%VENV%\Scripts\python" app.py --open
pause
