# SoftTwin — ソフト人工筋肉グリッパの Real↔Sim デジタルツイン

見た目が同じでも柔らかさが違う食品を，柔らかい指で**壊さずにつかむ**ための Physical AI システムです（Physical AI 講座 2026 応用編1 最終課題，isobe_97）。

## 何をするシステムか

空気圧で曲がるシリコン人工筋肉の指（3本）を持つグリッパを，Isaac Sim 上のデジタルツインと ROS 2 で同期させます。

1. 実機（M5Stack）の圧力と曲げ角の記録から，**人工筋肉モデル（圧力→曲げ角）を学習**します。
2. 学習したモデルをツインに入れて，**sim-to-real の誤差**を測ります。
3. 指の**たわみから把持力を推定**し，柔らかい物（いちご・豆腐・卵）を潰さずにつかみます。
4. 「何を，どれくらい優しくつかむか」は**言葉の指示から VLM** が決めます。

## 結果（MuJoCo，合成筋肉モデル，5物体×10シード，質量・摩擦・表面の柔らかさをランダム化）

| 手法 | 成功 | 破損 | 落下 | 力推定誤差 |
| --- | --- | --- | --- | --- |
| 開ループ 20 kPa 固定 | 78 % | 20 % | 2 % | 0.29 N |
| 開ループ 60 kPa 固定 | 60 % | 40 % | 0 % | 0.62 N |
| 閉ループ（線形アクチュエータモデル） | 70 % | 0 % | 30 % | 0.40 N |
| **閉ループ（学習モデル，提案）** | **96 %** | **0 %** | 2 % | **0.17 N** |

アクチュエータ同定（合成データ，評価16軌道）：曲げ角RMSE 線形 14.8° → 学習 2.2°（85 %減）．

### 触覚：見た目が同じで中身が違う物体（7物体×10シード）

指の内側に**布状触覚スキン（24点）**を貼ったと想定し，0.4 N だけ触って硬さを測ってから目標把持力を決め直します。

| 目標把持力の決め方 | 成功 | 破損 | 見た目が同じ物体の成功 | 1回の時間 |
| --- | --- | --- | --- | --- |
| VLM の事前知識のみ | 76 % | 14 % | 25 % | 11.7 s |
| 画像分類器 | 74 % | 19 % | 65 % | 14.6 s |
| 触覚分類器 | 87 % | 4 % | 80 % | 14.5 s |
| **画像＋触覚（提案）** | **91 %** | **4 %** | **90 %** | 13.9 s |
| 真値クラス（上限） | 93 % | 3 % | 85 % | 11.2 s |

分類器は，**人手なしで自動アノテーションしたデータ**（420試行，2 CPU で 4.6 分）で学習しました。各試行に，手先カメラの RGB・深度・マスク・バウンディングボックス，50 Hz の内圧・関節角・推定／真の把持力・スキン24点（センサモデル後／真値），隠れた物理量，成否（成功・破損・滑落）が入っています（`data/tactile_dataset/DATASET_CARD.md`）。

再現コマンドです。

```text
python sim_mujoco/softtwin_mj.py bench --seeds 10
python sim_mujoco/tactile_dataset.py generate --per-object 60
python sim_mujoco/tactile_dataset.py train
python sim_mujoco/tactile_dataset.py bench --seeds 10
python scripts/make_figures.py && python scripts/make_tactile_figures.py
python sim_mujoco/make_demo_video.py
```

レポートは `docs/paper/説明資料_PAI最終課題_isobe_97.pdf`，動画は `media/softtwin_demo.mp4` です。

## 構成

| 層 | 役割 | 場所 |
| --- | --- | --- |
| 実機 | 人工筋肉グリッパ。M5Stack が micro-ROS で駆動と計測を担う | `firmware/m5stack_softgripper` |
| 仮想実機 | ハードがないときに M5Stack を代替（同じトピック） | `core/virtual_m5stack.py` |
| シミュレーション（MuJoCo） | 評価用のセル。接触力の真値で推定器を検証 | `sim_mujoco/softtwin_mj.py` |
| シミュレーション（Isaac Sim） | Isaac Sim のセル（ガントリ＋3本指＋物体＋トレー）。PRBM で柔軟指を近似 | `sim/softtwin_scene.py` |
| ランタイム | アクチュエータモデル，把持エピソード，固定シードのベンチマーク | `sim/softtwin_runtime.py` |
| ROS 2 グラフ | `/clock`，`/softtwin/joint_states`，`/softtwin/joint_command`，カメラ | `sim/ros2_graph.py` |
| コアエンジン | 実機とツインの同期，誤差・遅延の計測，RabbitMQ への発行 | `core/engine.py` |
| 学習 | 人工筋肉モデルの同定（numpy MLP）と線形モデルとの比較 | `learning/actuator_net.py` |
| 触覚 | 布状触覚スキンのセンサモデル，触覚プローブの特徴量，分類器 | `sim_mujoco/tactile.py` |
| データ生成 | 自動アノテーション付きマルチモーダルデータの量産，学習，評価 | `sim_mujoco/tactile_dataset.py` |
| 触覚（Isaac Sim） | PhysX の接触レポートから同じ24点のスキンを作る（実機PCで未実行） | `sim/tactile_isaac.py` |
| アプリ層 | 言葉の指示＋画像 → 対象と把持力クラス（Ollama VLM，予備にキーワード） | `app/instruction.py` |
| UI | HTTP＋WebSocket ゲートウェイと監視ダッシュボード | `core/ui_server.py`, `ui/dashboard.html` |
| MCP | LLM エージェント（MCP クライアント）から Isaac Sim を操作するブリッジ（Kit 拡張＋MCP サーバ） | `isaac_ext/pai.mcp.bridge`, `mcp_server/isaacsim_mcp.py` |

## セットアップ（Windows 11）

1. PowerShell で `powershell -ExecutionPolicy Bypass -File tools\bootstrap.ps1` を実行します。Isaac Sim 5.0（standalone）と ROS 2 Jazzy（pixi，`C:\pai_ros`）を準備します。
2. MCP クライアント（LLM エージェント）を使う場合は，`tools\bootstrap.ps1` が出力する MCP 設定スニペットをクライアントの設定に追加します。
3. `tools\launch_sim.bat` を実行します。Zenoh ルータと Isaac Sim（MCP ブリッジと ROS 2 ブリッジを有効化した状態）が起動します。

## 動かし方（MCP 経由／手動とも同じ）

```text
isaac_run_file("sim/softtwin_scene.py")     # セルを構築
isaac_run_file("sim/ros2_graph.py")         # ROS 2 グラフ
isaac_run_file("sim/softtwin_runtime.py")   # ランタイムを読み込み
isaac_exec("await run_episode('strawberry', mode='closed_loop'); _result = LAST")
isaac_exec("await benchmark(seeds=range(10)); _result = SUMMARY")
```

ROS 2 側のコマンドです。それぞれ別のターミナルで実行します。

```text
pixi run --manifest-path C:\pai_ros\pixi.toml virtual_hw
pixi run --manifest-path C:\pai_ros\pixi.toml core
pixi run --manifest-path C:\pai_ros\pixi.toml ui
```

UI は http://localhost:8080/ui/dashboard.html で開きます。

## 制約と今後の課題

- `learning/` の初期データは **合成データ（SyntheticMuscle）**です。実機ログ（`--csv data/real_log_*.csv`）での再学習は今後の課題です。
- 把持力は**推定値**（関節のたわみ×曲げ剛性÷レバー長）です。真値との比較（PhysX の接触レポート，または実機の荷重計測）は今後の課題です。
- 物体の位置は，現状シミュレーションの真値を使っています。カメラによる検出に置き換えるのが次の段階です。
- 物体の柔らかさは MuJoCo の接触パラメータで近似しています（有限要素の変形ではありません）。触覚スキンの特性も仮定値です。
