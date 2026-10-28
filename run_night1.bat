@echo off
rem Night run 1: 1e8 uniform draws, 20 blocks of 250 seeds, 8 workers, seeds from 10000. Rerun this file to resume after a stop.
cd /d "C:\Users\jwncr\Claude Code Test\hyperchaos-catalogue"
set PYTHONPATH=src
echo Started %date% %time% >> results\h2_night1.log
.venv\Scripts\python.exe -u experiments\h2_night.py night1 8 20 250 10000 >> results\h2_night1.log 2>&1
echo Finished %date% %time% exit code %errorlevel% >> results\h2_night1.log
