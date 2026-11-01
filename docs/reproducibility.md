# 再現の手順

## 環境

| 項目 | 確認した版 |
| --- | --- |
| OS | Ubuntu 24.04（CI と MuJoCo）、Windows 11（Isaac Sim） |
| Python | 3.12（CI）、3.13 |
| MuJoCo | 3.15.0（`requirements.txt` で固定） |
| Isaac Sim | 5.0.0 standalone（Windows、Kit 107.3、描画なし） |

GPU は要らない。MuJoCo の描画（データ生成と動画）は、Linux では OSMesa か EGL を使う（`MUJOCO_GL=osmesa`。`sudo apt install libosmesa6` が要る）。

## 結果から数字を作り直す（約 10 秒）

```text
python scripts/analysis_stats.py      # results/*.json -> results/metrics.json（区間・検定・内訳・要求）
python scripts/make_numbers.py        # metrics.json -> docs/paper/numbers.tex と README.md
python scripts/make_figures.py        # 図（docs/figures/）
```

## すべての実験を最初から（2 CPU で約 1.5 時間）

| コマンド | 時間の目安 | 出力 |
| --- | --- | --- |
| `python learning/actuator_net.py --synthetic` | 約 1 分 | `learning/actuator_net.json`、`results/actuator_identification.json` |
| `python scripts/fit_greybox.py` | 約 1 分 | `learning/actuator_greybox.json`、`learning/actuator_greybox_play.json`（遊び要素つき） |
| `python scripts/actuator_compare.py` | 約 1 分 | `results/actuator_compare.json`（RMSE と把持中の偏り） |
| `python scripts/calibrate_contacts.py` | 約 5 分 | `sim_mujoco/contact_calibration.json`（接触剛性を N/m に合わせる） |
| `python sim_mujoco/softtwin_mj.py recipe` | 約 1 分 | `results/open_loop_recipe.json`（区分ごとの内圧表、シード 10000） |
| `python sim_mujoco/softtwin_mj.py bench --seeds 30` | 約 10 分 | `results/benchmark_mujoco.json`（Q1） |
| `python sim_mujoco/softtwin_mj.py bench --seeds 30 --seed0 30 --out results/benchmark_mujoco_holdout.json` | 約 10 分 | 開発に使っていないシードでの Q1 |
| `python scripts/estimator_error_split.py` | 約 5 分 | `results/estimator_error_split.json`（推定値÷真値の分解） |
| `python scripts/control_analysis.py` | 約 1 分 | `results/control_analysis.json` |
| `python scripts/robustness.py` | 約 10 分 | `results/robustness.json` |
| `python scripts/size_shift.py` | 約 6 分 | `results/size_shift.json`（大きさを 0.8・1.2 倍にした物体） |
| `python scripts/probe_sensitivity.py` | 約 5 分 | `results/probe_sensitivity.json`、判別マップ |
| `python sim_mujoco/tactile_dataset.py generate --per-object 60` | 約 5 分 | `data/tactile_dataset/`（Q3） |
| `python scripts/validate_dataset.py` | 数秒 | 検証結果 |
| `python sim_mujoco/tactile_dataset.py train && python sim_mujoco/tactile_dataset.py train-loo` | 数秒 | 分類器と力の窓 |
| `python sim_mujoco/tactile_dataset.py bench --seeds 30` | 約 15 分 | `results/benchmark_lookalike.json`（Q2） |
| `python sim_mujoco/tactile_dataset.py bench --seeds 30 --seed0 30 --out results/benchmark_lookalike_holdout.json` | 約 15 分 | 開発に使っていないシードでの Q2 |
| `python sim_mujoco/media.py` | 約 10 分 | `media/*.gif`、`media/*.mp4`、図 1 |
| `python sim_mujoco/narration.py` | 約 1 分 | `media/narration/*.wav`（ナレーション。VOICEVOX engine 0.24 を `127.0.0.1:50021` で起動しておく。声は No.7、読みの確認は `--check`） |
| `python sim_mujoco/make_video.py` | 約 6 分 | `media/softtwin_demo.mp4`（提出動画、ナレーションと BGM 入り）と章の時刻 |

## PhysX（Isaac Sim 5.0、Windows、約 3 時間）

```text
set ISAACSIM=C:\isaac-sim-standalone-5.0.0-windows-x86_64
sim\isaac_headless\run_headless.bat        # isaac_io\v2_rows.jsonl（1 試行 1 行、止まっても続きから）
python scripts/collect_isaac.py isaac_io   # -> results/benchmark_isaac.json、results/isaac_replay/
```

計画は `isaac_io\v2_plan.json`（シード、物体、方式、接触剛性の確認、再生用に姿勢を記録する試行）で変えられる。
描画を使わない Kit アプリ（`sim/isaac_headless/softtwin_physics.kit`）で PhysX だけを動かすので、RTX の描画が動かない GPU ドライバでも走る。

## 決定性

同じ環境なら、同じシードの試行はビット単位まで同じ結果になる。

```text
python sim_mujoco/softtwin_mj.py bench --seeds 2 --out /tmp/q1.json
python scripts/check_determinism.py /tmp/q1.json      # 公開した 80 行と一致するか
```
