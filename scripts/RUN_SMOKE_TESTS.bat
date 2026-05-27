@echo off
cd /d "%~dp0"
echo Running py_compile checks...
python -m py_compile shared\core\app_metadata.py shared\core\app_prefs.py shared\core\support_bundle.py shared\core\mass_ai_domain.py shared\core\mass_ai_engine.py shared\core\ops_store.py old_desktop\mass_ai_desktop.py old_desktop\ui_kit.py new_web\dashboard\app.py MASS_AI_LAUNCHER.py
if errorlevel 1 goto :fail
echo.
echo Running unit tests...
python -m unittest discover -s shared\tests -t shared -p "test_*.py" -v
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
