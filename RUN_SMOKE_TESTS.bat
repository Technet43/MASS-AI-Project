@echo off
cd /d "%~dp0"
echo Running py_compile checks...
python -m py_compile project\core\app_metadata.py project\core\support_bundle.py project\core\mass_ai_domain.py project\core\mass_ai_engine.py project\old_desktop\mass_ai_desktop.py project\old_desktop\ui_kit.py project\new_web\dashboard\app.py MASS_AI_LAUNCHER.py
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
