@echo off
setlocal
cd /d "%~dp0"
rem Prefer Python on this PC's PATH, including an activated virtual environment.
python -c "import sys; sys.exit(0 if sys.version_info >= (3,12) and sys.maxsize > 2**32 else 1)" >nul 2>&1
if not errorlevel 1 goto run_python
rem Fallback for Windows installations with the py launcher but no python on PATH.
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3,12) and sys.maxsize > 2**32 else 1)" >nul 2>&1
if not errorlevel 1 goto run_py
echo Python 3.12+ 64-bit was not found. Install Python or select its interpreter in VS Code.
pause
exit /b 1

:run_python
python -B -X utf8 "%~dp0v3.5.py" %*
goto done

:run_py
py -3 -B -X utf8 "%~dp0v3.5.py" %*
goto done

:done
set "D435_EXIT=%errorlevel%"
if not "%D435_EXIT%"=="0" pause
exit /b %D435_EXIT%
