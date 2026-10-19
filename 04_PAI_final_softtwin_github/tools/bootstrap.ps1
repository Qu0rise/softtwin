# SoftTwin bootstrap (run once, from a normal PowerShell window):
#   powershell -ExecutionPolicy Bypass -File "<repo>\tools\bootstrap.ps1"
# What it does (idempotent, safe to re-run):
#   1. Writes an environment report (GPU / VRAM / driver / RAM / disk / tools)
#   2. Downloads Isaac Sim 6.1.0 standalone (Windows) and extracts it to C:\isaacsim
#   3. Installs pixi (winget) and a ROS 2 Jazzy environment in C:\pai_ros (short path)
#   4. Self-tests rclpy + mcp, writes the Claude Desktop MCP config snippet
# Logs: <repo>\tools\logs\bootstrap_*.log and env_report.json
param(
    [string]$IsaacDir = "C:\isaacsim",
    [string]$RosDir = "C:\pai_ros",
    [string]$DownloadDir = "$env:USERPROFILE\Downloads",
    [switch]$SkipIsaac
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$Repo = Split-Path -Parent $PSScriptRoot
$LogDir = Join-Path $Repo "tools\logs"
New-Item -ItemType Directory -Force $LogDir | Out-Null
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
Start-Transcript -Path (Join-Path $LogDir "bootstrap_$ts.log") | Out-Null

$IsaacZipName = "isaac-sim-standalone-6.1.0-windows-x86_64.zip"
$IsaacUrl = "https://downloads.isaacsim.nvidia.com/$IsaacZipName"
$IsaacMd5 = "a07968e980072c9ca27b2166443e2d89"

$report = [ordered]@{ time = (Get-Date).ToString("s"); repo = $Repo; isaac_dir = $IsaacDir; ros_dir = $RosDir; steps = [ordered]@{} }
function Save-Report { $report | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 (Join-Path $LogDir "env_report.json") }
function Step([string]$name, [scriptblock]$body) {
    Write-Host "`n=== $name" -ForegroundColor Cyan
    try { $r = & $body; $report.steps[$name] = [ordered]@{ ok = $true; info = $r } }
    catch { $report.steps[$name] = [ordered]@{ ok = $false; error = "$_" }; Write-Warning "$name failed: $_" }
    Save-Report
}
function Has($cmd) { [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }

Step "environment" {
    $gpu = $null
    if (Has "nvidia-smi") { $gpu = (& nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader) -join "; " }
    $os = (Get-CimInstance Win32_OperatingSystem)
    $disks = Get-CimInstance Win32_LogicalDisk -Filter "DriveType=3" | ForEach-Object { "{0} free {1:N0} GB / {2:N0} GB" -f $_.DeviceID, ($_.FreeSpace / 1GB), ($_.Size / 1GB) }
    $lp = (Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -ErrorAction SilentlyContinue).LongPathsEnabled
    [ordered]@{
        gpu = $gpu; os = "$($os.Caption) $($os.Version)"; ram_gb = [math]::Round($os.TotalVisibleMemorySize / 1MB, 1)
        disks = $disks; long_paths_enabled = $lp
        git = Has "git"; winget = Has "winget"; pixi = Has "pixi"; ollama = Has "ollama"; docker = Has "docker"; node = Has "node"; python = Has "python"
        cpu = (Get-CimInstance Win32_Processor | Select-Object -First 1).Name
    }
}

# pick a drive with enough room if the default one is short (zip ~10 GB + extracted)
if (-not (Test-Path (Join-Path $IsaacDir "isaac-sim.bat"))) {
    $need = 60
    $freeC = (Get-PSDrive -Name ($IsaacDir.Substring(0, 1))).Free / 1GB
    if ($freeC -lt $need) {
        $alt = Get-PSDrive -PSProvider FileSystem | Where-Object { $_.Free / 1GB -ge $need } | Sort-Object Free -Descending | Select-Object -First 1
        if ($alt) { $IsaacDir = "$($alt.Name):\isaacsim"; $DownloadDir = "$($alt.Name):\isaacsim_download"; Write-Host "Using $IsaacDir (more free space)" }
    }
    $report.isaac_dir = $IsaacDir
}

if (-not $SkipIsaac) {
    Step "isaac_sim_download" {
        if (Test-Path (Join-Path $IsaacDir "isaac-sim.bat")) { return "already installed at $IsaacDir" }
        $free = (Get-PSDrive -Name ($IsaacDir.Substring(0, 1))).Free / 1GB
        if ($free -lt 60) { throw ("only {0:N0} GB free on the Isaac Sim drive; need ~60 GB (zip + extracted). Re-run with -IsaacDir D:\isaacsim" -f $free) }
        New-Item -ItemType Directory -Force $DownloadDir | Out-Null
        $zip = Join-Path $DownloadDir $IsaacZipName
        Write-Host "Downloading $IsaacUrl (several GB, resumable) ..."
        & curl.exe -L --fail --retry 5 --retry-delay 5 -C - -o $zip $IsaacUrl
        if ($LASTEXITCODE -ne 0) { throw "curl failed ($LASTEXITCODE)" }
        $md5 = (Get-FileHash $zip -Algorithm MD5).Hash.ToLower()
        if ($md5 -ne $IsaacMd5) { throw "MD5 mismatch: $md5 (expected $IsaacMd5). Delete $zip and re-run." }
        "downloaded $zip md5 ok"
    }
    Step "isaac_sim_extract" {
        if (Test-Path (Join-Path $IsaacDir "isaac-sim.bat")) { return "already extracted" }
        $zip = Join-Path $DownloadDir $IsaacZipName
        New-Item -ItemType Directory -Force $IsaacDir | Out-Null
        Write-Host "Extracting to $IsaacDir (takes a while) ..."
        & tar.exe -xf $zip -C $IsaacDir
        if ($LASTEXITCODE -ne 0) { throw "tar extract failed ($LASTEXITCODE)" }
        if (-not (Test-Path (Join-Path $IsaacDir "isaac-sim.bat"))) {
            $inner = Get-ChildItem $IsaacDir -Directory | Where-Object { Test-Path (Join-Path $_.FullName "isaac-sim.bat") } | Select-Object -First 1
            if ($inner) { Get-ChildItem $inner.FullName -Force | Move-Item -Destination $IsaacDir; Remove-Item $inner.FullName -Recurse -Force }
        }
        if (-not (Test-Path (Join-Path $IsaacDir "isaac-sim.bat"))) { throw "isaac-sim.bat not found after extraction" }
        "extracted"
    }
    Step "isaac_sim_post_install" {
        $pi = Join-Path $IsaacDir "post_install.bat"
        if (Test-Path $pi) { Push-Location $IsaacDir; & cmd /c $pi; Pop-Location; "post_install.bat ran (exit $LASTEXITCODE)" } else { "no post_install.bat" }
    }
}

Step "pixi" {
    if (-not (Has "pixi")) {
        if (-not (Has "winget")) { throw "winget not found; install pixi manually from https://pixi.sh" }
        & winget install -e --id prefix-dev.pixi --accept-source-agreements --accept-package-agreements
        $env:Path = [Environment]::GetEnvironmentVariable("Path", "User") + ";" + [Environment]::GetEnvironmentVariable("Path", "Machine") + ";$env:USERPROFILE\.pixi\bin"
    }
    if (-not (Has "pixi")) { throw "pixi still not on PATH; open a new PowerShell and re-run" }
    (& pixi --version) -join ""
}

Step "ros2_env" {
    New-Item -ItemType Directory -Force $RosDir | Out-Null
    $toml = Get-Content (Join-Path $Repo "pixi_ros\pixi.toml") -Raw
    $toml = $toml.Replace("__PAI_REPO__", ($Repo -replace "\\", "/"))
    if ($IsaacDir -ne "C:\isaacsim") { $toml = $toml.Replace('C:\\isaacsim', ($IsaacDir -replace "\\", "\\")) }
    Set-Content -Encoding UTF8 (Join-Path $RosDir "pixi.toml") $toml
    & pixi install --manifest-path (Join-Path $RosDir "pixi.toml")
    if ($LASTEXITCODE -ne 0) { throw "pixi install failed ($LASTEXITCODE)" }
    "ROS 2 Jazzy env ready in $RosDir"
}

Step "selftest" {
    $out = & pixi run --manifest-path (Join-Path $RosDir "pixi.toml") python -c "import rclpy, mcp, numpy; print('rclpy+mcp ok')" 2>&1
    ($out | Select-Object -Last 3) -join " | "
}

Step "mcp_launcher" {
    $pixiExe = (Get-Command pixi).Source
    $bat = "@echo off`r`nset PAI_REPO=$Repo`r`n`"$pixiExe`" run --manifest-path `"$(Join-Path $RosDir 'pixi.toml')`" python `"$(Join-Path $Repo 'mcp_server\isaacsim_mcp.py')`"`r`n"
    Set-Content -Encoding ASCII (Join-Path $RosDir "run_mcp.bat") $bat -NoNewline
    "wrote $(Join-Path $RosDir 'run_mcp.bat')"
}

Step "mcp_config_snippet" {
    $pixiExe = (Get-Command pixi).Source
    $cfg = [ordered]@{ mcpServers = [ordered]@{ isaacsim = [ordered]@{
                command = $pixiExe
                args    = @("run", "--manifest-path", (Join-Path $RosDir "pixi.toml"), "python", (Join-Path $Repo "mcp_server\isaacsim_mcp.py"))
                env     = [ordered]@{ PAI_REPO = $Repo }
            } } }
    $path = Join-Path $Repo "tools\claude_desktop_mcp_snippet.json"
    $cfg | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 $path
    "wrote $path"
}

Step "ollama_vlm" {
    if (-not (Has "ollama")) { return "ollama not installed (VLM falls back to keyword parser)" }
    & ollama pull qwen2.5vl:7b
    "ollama pull qwen2.5vl:7b exit $LASTEXITCODE"
}

$report.finished = (Get-Date).ToString("s")
Save-Report
Write-Host "`nDone. Report: $LogDir\env_report.json" -ForegroundColor Green
Write-Host "Next: add tools\claude_desktop_mcp_snippet.json to Claude Desktop (Settings > Developer > Edit Config), restart Claude, then run tools\launch_sim.bat"
Stop-Transcript | Out-Null
