@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" launch.py
) else (
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3 launch.py
    ) else (
        where python >nul 2>nul
        if not errorlevel 1 (
            python launch.py
        ) else (
            echo Install Python 3.10 or later from python.org, then open this file again.
            pause
            exit /b 1
        )
    )
)
if errorlevel 1 (
    pause
    exit /b 1
)
