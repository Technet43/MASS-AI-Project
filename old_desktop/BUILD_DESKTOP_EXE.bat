@echo off
setlocal
cd /d "%~dp0"
echo Running compile checks before packaging...
python -m py_compile ..\shared\core\app_metadata.py ..\shared\core\support_bundle.py ..\shared\core\mass_ai_domain.py ..\shared\core\mass_ai_engine.py ..\shared\core\ops_store.py mass_ai_desktop.py ui_kit.py ..\MASS_AI_LAUNCHER.py
if errorlevel 1 goto :fail
echo.
echo Running unit tests before packaging...
python -m unittest discover -s ..\shared\tests -t ..\shared -p "test_*.py" -v
if errorlevel 1 goto :fail
echo.
python -m pip install pyinstaller
python -m PyInstaller --noconfirm --clean --onefile --windowed --name MASS_AI_Desktop --distpath artifacts\dist --workpath artifacts\build --specpath packaging --paths ..\shared\core --collect-all xgboost --collect-all sklearn --collect-data matplotlib mass_ai_desktop.py
python -c "import sys, json; from pathlib import Path; sys.path.insert(0, str(Path('..\\shared\\core').resolve())); from app_metadata import metadata_dict; Path('artifacts/dist').mkdir(parents=True, exist_ok=True); Path('artifacts/dist/build_manifest.json').write_text(json.dumps(metadata_dict(), indent=2), encoding='utf-8')"
echo.
echo The Windows executable will be created here:
echo %~dp0artifacts\dist
pause
exit /b 0

:fail
echo.
echo Packaging aborted because compile or test checks failed.
pause
exit /b 1
