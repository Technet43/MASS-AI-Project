@echo off
cd /d "%~dp0"
echo Running py_compile checks...
python -m py_compile project\app_metadata.py project\support_bundle.py project\mass_ai_domain.py project\mass_ai_engine.py project\mass_ai_desktop.py MASS_AI_LAUNCHER.py
if errorlevel 1 goto :fail
echo.
echo Running unit tests...
python -m unittest discover -s project\tests -t project -p "test_*.py" -v
if errorlevel 1 goto :fail
echo.
echo Smoke tests passed.
pause
exit /b 0

:fail
echo.
echo Smoke tests failed.
pause
exit /b 1
