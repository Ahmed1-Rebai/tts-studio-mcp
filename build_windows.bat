@echo off
REM Build the TTS app into a single Windows .exe with PyInstaller.
REM Requirements: Python 3.8+ installed, and an internet connection.
setlocal

echo ============================================
echo   TTS App - Windows build
echo ============================================

echo.
echo [1/3] Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    if errorlevel 1 goto error
)

echo [2/3] Installing dependencies...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt pyinstaller
if errorlevel 1 goto error

echo [3/3] Building executable (this can take a minute)...
pyinstaller --noconfirm --clean tts_app.spec
if errorlevel 1 goto error

echo.
echo ============================================
echo Build complete!
echo Your app is here:  dist\TTSApp.exe
echo Copy dist\TTSApp.exe to a friend's computer
echo and run it - no installation needed.
echo ============================================
pause
exit /b 0

:error
echo.
echo Build failed. Please check the errors above.
pause
exit /b 1