@echo off
title Pokemon RL Battle Agent - Demo Startup
cls

echo ===========================================================
echo    Pokemon RL Battle Agent - Demo Startup Flow
echo ===========================================================
echo.
echo [1/3] Preparing Model Directories...
if not exist "models\ppo" mkdir "models\ppo"
if not exist "models\dqn" mkdir "models\dqn"

echo.
echo [2/3] Checking Showdown Server...
echo To run the demo, the Pokemon Showdown server MUST be running.
echo Please ensure you have run 'node pokemon-showdown' in the 'showdown' folder.
echo.
echo [3/3] Launching Agent Selection Loop...
echo.

:loop
echo -----------------------------------------------------------
echo   COMMANDS:
echo   1. Start Human Battle (Agent vs Human)
echo   2. Run Evaluation (PPO vs DQN Control)
echo   3. Run Iterative Training (25k Steps)
echo   4. Exit
echo -----------------------------------------------------------
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo Launching Human Battle...
    python\venv\Scripts\python.exe python\battle_vs_human.py
    goto loop
)
if "%choice%"=="2" (
    echo Launching PPO vs DQN Evaluation...
    python\venv\Scripts\python.exe python\battle_control.py
    goto loop
)
if "%choice%"=="3" (
    echo Starting Standard Retraining...
    echo Sequential execution: PPO then DQN.
    echo [PPO Training]
    python\venv\Scripts\python.exe python\train_ppo.py
    echo [DQN Training]
    python\venv\Scripts\python.exe python\train_dqn.py
    echo Retraining complete.
    pause
    goto loop
)
if "%choice%"=="4" (
    exit
)

echo Invalid choice. Please try again.
pause
goto loop
