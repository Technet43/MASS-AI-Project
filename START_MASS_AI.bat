@echo off
cd /d "%~dp0\project\web"
python -m streamlit run dashboard\app.py
