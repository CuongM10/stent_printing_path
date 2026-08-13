@echo off
cd /d %~dp0
python -m pip install -r requirements.txt
python stent_path_optimizer.py
pause
