@echo off
cd /d "%~dp0\project\new_web"
python -m streamlit run dashboard\app.py
