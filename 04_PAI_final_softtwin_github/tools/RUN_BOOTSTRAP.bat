@echo off
rem Double-click to run the SoftTwin bootstrap (Isaac Sim 6.1.0 + ROS 2 Jazzy via pixi). Safe to re-run.
title SoftTwin bootstrap
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0bootstrap.ps1"
echo.
echo Bootstrap finished. This window can be closed.
pause
