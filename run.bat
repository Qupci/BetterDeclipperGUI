@echo off
rem Starts BetterDeclipper GUI in the browser (install.bat sets it up). Options: run.bat --help
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto notinstalled
".venv\Scripts\python.exe" -m betterdeclipper_gui %*
if errorlevel 1 pause
exit /b

:notinstalled
echo BetterDeclipper GUI is not installed yet: run install.bat first.
pause
exit /b 1
