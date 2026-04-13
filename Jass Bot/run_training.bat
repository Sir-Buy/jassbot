@echo off
cd /d "%~dp0"
python -m bot.train_v6 --epochs 5000 --resume checkpoints_v6/latest.pt --ent-end 0.01
pause
