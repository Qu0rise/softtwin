@echo off
rem Starts the Zenoh router (ROS 2 RMW) in its own window, then Isaac Sim with the MCP bridge + ROS 2 bridge.
setlocal
set PIXI=%USERPROFILE%\.pixi\bin\pixi.exe
if not exist "%PIXI%" set PIXI=pixi
set MANIFEST=C:\pai_ros\pixi.toml
start "zenoh-router" "%PIXI%" run --manifest-path %MANIFEST% zenoh
timeout /t 3 /nobreak >nul
"%PIXI%" run --manifest-path %MANIFEST% sim
endlocal
