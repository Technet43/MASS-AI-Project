@echo off
setlocal
cd /d "%~dp0"
echo Running compile checks before packaging...
python -m py_compile project\app_metadata.py project\support_bundle.py project\mass_ai_domain.py project\mass_ai_engine.py project\mass_ai_desktop.py MASS_AI_LAUNCHER.py
if errorlevel 1 goto :fail
echo.
echo Running unit tests before packaging...
python -m unittest discover -s project\tests -t project -p "test_*.py" -v
if errorlevel 1 goto :fail
echo.
cd /d "%~dp0\project"
python -m pip install pyinstaller
python -m PyInstaller --noconfirm --clean --onefile --windowed --name MASS_AI_Desktop --distpath artifacts\dist --workpath artifacts\build --specpath packaging --collect-all xgboost --collect-all sklearn --collect-data matplotlib mass_ai_desktop.py
python -c "from pathlib import Path; import json; from app_metadata import metadata_dict; Path('artifacts/dist').mkdir(parents=True, exist_ok=True); Path('artifacts/dist/build_manifest.json').write_text(json.dumps(metadata_dict(), indent=2), encoding='utf-8')"
echo.
echo The Windows executable will be created here:
echo %~dp0project\artifacts\dist
pause
exit /b 0

:fail
echo.
echo Packaging aborted because compile or test checks failed.
pause
exit /b 1
