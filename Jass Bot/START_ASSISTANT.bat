@echo off
title Jass Bot - V7 RL Engine (44%% win rate)
cd /d "%~dp0"

echo.
echo  ============================================
echo    JASS DIFFERENZLER BOT - V7 RL ENGINE
echo  ============================================
echo.
echo  Engine:  V7 League-Trained Neural Net
echo  Tested:  44%% win rate in mixed field
echo  Decl:    Expert formula
echo  Server:  http://localhost:5000
echo.
echo  MODES:
echo    WATCH  = observe only
echo    ASSIST = bot recommends, you play
echo    BOT    = full auto-play
echo.
echo  Starting server...
echo.

python server.py --port 5000 --mode rl

pause
