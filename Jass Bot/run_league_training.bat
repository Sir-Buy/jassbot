@echo off
cd /d "%~dp0"
python -m bot.train_v7_league --resume-v6 checkpoints_v6/best.pt --epochs 10000
pause
