@echo off
cd /d %~dp0
python -m pip install -r requirements.txt
python stent_path_optimizer.py --all --diameter 8 --length 20 --candidates 500
pause
