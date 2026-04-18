@echo off
setlocal
cd /d "%~dp0"
echo Running compile checks before packaging...
python -m py_compile project\core\app_metadata.py project\core\support_bundle.py project\core\mass_ai_domain.py project\core\mass_ai_engine.py project\old_desktop\mass_ai_desktop.py project\old_desktop\ui_kit.py MASS_AI_LAUNCHER.py
if errorlevel 1 goto :fail
echo.
echo Running unit tests before packaging...
python -m unittest discover -s project\tests -t project -p "test_*.py" -v
if errorlevel 1 goto :fail
echo.
cd /d "%~dp0\project\old_desktop"
python -m pip install pyinstaller
python -m PyInstaller --noconfirm --clean --onefile --windowed --name MASS_AI_Desktop --distpath artifacts\dist --workpath artifacts\build --specpath packaging --paths ..\core --collect-all xgboost --collect-all sklearn --collect-data matplotlib mass_ai_desktop.py
python -c "import sys, json; from pathlib import Path; sys.path.insert(0, str(Path('..\\core').resolve())); from app_metadata import metadata_dict; Path('artifacts/dist').mkdir(parents=True, exist_ok=True); Path('artifacts/dist/build_manifest.json').write_text(json.dumps(metadata_dict(), indent=2), encoding='utf-8')"
echo.
echo The Windows executable will be created here:
echo %~dp0project\old_desktop\artifacts\dist
pause
exit /b 0

:fail
echo.
echo Packaging aborted because compile or test checks failed.
pause
exit /b 1
