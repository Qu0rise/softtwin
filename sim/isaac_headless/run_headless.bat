@echo off
rem Q1 in Isaac Sim 5.0's PhysX without the RTX renderer.
rem   ISAACSIM : Isaac Sim 5.0 standalone folder (default C:\isaac-sim-standalone-5.0.0-windows-x86_64)
rem   plan     : isaac_io\v2_plan.json (optional; seeds, objects, configs, contact check, runs to record for replay)
rem   output   : isaac_io\v2_rows.jsonl (one line per episode, resumable), v2_status.json, v2_summary.json
if "%ISAACSIM%"=="" set ISAACSIM=C:\isaac-sim-standalone-5.0.0-windows-x86_64
pushd "%~dp0..\.."
set PAI_REPO=%CD%
if not exist isaac_io mkdir isaac_io
"%ISAACSIM%\kit\kit.exe" "%PAI_REPO%\sim\isaac_headless\softtwin_physics.kit" --no-window ^
  --ext-folder "%ISAACSIM%\exts" --ext-folder "%ISAACSIM%\extscache" --ext-folder "%ISAACSIM%\extsDeprecated" ^
  --exec "%PAI_REPO%\sim\isaac_headless\q1_isaac.py" --/log/file="%PAI_REPO%\isaac_io\headless.log" --/log/fileLogLevel=warn
popd
