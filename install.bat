@echo off
rem BetterDeclipper GUI installer: sets up everything in the .venv folder next to this file.
rem torch comes as the CUDA build when an NVIDIA GPU is present (about 2.5 GB), else the CPU build.
setlocal
cd /d "%~dp0"
title BetterDeclipper GUI - install

rem Python 3.10+ from python.org or the Microsoft Store (MSYS2 / MinGW builds cannot install torch)
set "CHECK=import sys; sys.exit(not (sys.version_info >= (3, 10) and 'MSC' in sys.version))"
set "BASE="
for %%P in ("py -3.12" "py -3.11" "py -3.13" "py -3.10" "py -3" "python") do (
  if not defined BASE (
    %%~P -c "%CHECK%" >nul 2>nul && set "BASE=%%~P"
  )
)
if not defined BASE goto nopython
for /f "delims=" %%V in ('%BASE% --version 2^>^&1') do echo Using %%V

if exist ".venv\Scripts\python.exe" goto havevenv
echo Creating the Python environment in .venv ...
%BASE% -m venv .venv
if errorlevel 1 goto failed
if not exist ".venv\Scripts\python.exe" goto failed

:havevenv
set "PY=%~dp0.venv\Scripts\python.exe"
"%PY%" -m pip install --upgrade pip
if errorlevel 1 goto failed

"%PY%" -c "import torch" >nul 2>nul && goto havetorch
where nvidia-smi >nul 2>nul && goto cudatorch
echo.
echo No NVIDIA GPU found: installing the CPU build of torch, about 250 MB ...
"%PY%" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto failed
goto havetorch

:cudatorch
echo.
echo NVIDIA GPU found: installing the CUDA build of torch, about 2.5 GB ...
"%PY%" -m pip install torch --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 goto failed

:havetorch
echo.
rem pip keeps an installed declipper as it is: when updating, get its latest version as well
set "UPDATE="
"%PY%" -m pip show betterdeclipper >nul 2>nul && set "UPDATE=1"
echo Installing BetterDeclipper and the GUI ...
"%PY%" -m pip install -e .
if errorlevel 1 goto failed
if not defined UPDATE goto installed
echo.
echo Updating BetterDeclipper to its latest version ...
"%PY%" -m pip install --force-reinstall --no-deps "betterdeclipper @ https://github.com/Qupci/BetterDeclipper/archive/refs/heads/main.zip"
if errorlevel 1 goto failed

:installed
echo.
echo Done. Start the app with run.bat
pause
exit /b 0

:nopython
echo Python 3.10 or newer was not found.
echo Install it from https://www.python.org/downloads/ and tick "Add python.exe to PATH",
echo then run install.bat again.
pause
exit /b 1

:failed
echo.
echo The installation failed, see the messages above.
pause
exit /b 1
