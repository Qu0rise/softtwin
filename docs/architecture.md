# 構成と入出力

コードの入口と、モジュールのあいだで受け渡す量の単位・周期・意味をまとめる。
物理パラメータの根拠は [physical_parameters.md](physical_parameters.md)、評価の数え方は [evaluation_protocol.md](evaluation_protocol.md) にある。

## 1. 一つの制御コア、二つの物理エンジン

```
sim_mujoco/twin_core.py      物体と個体の乱数、模擬筋肉、筋肉のモデル、センサ、推定器、制御器、把持の手順、成否の判定
        │  engine インタフェース（下の表）だけを使う
        ├── sim_mujoco/softtwin_mj.py          MuJoCo のセル（MJCF を生成）
        ├── sim/isaac_headless/q1_isaac.py     Isaac Sim 5.0 / PhysX のセル（USD を生成、cell_usd.py）
        └── core/grip_controller.py            実機用（ROS 2 ノード ros2/softtwin_grasp_node.py から呼ぶ）
```

2 つのエンジンの結果の違いは、物理（接触と関節の解き方）からしか生まれない。制御側のコードは 1 つなので、そこに差が入る余地がない。

| engine の属性・関数 | 単位 | 意味 |
| --- | --- | --- |
| `dt_ctrl` | s | 制御周期（0.02） |
| `t` | s | シミュレーション時刻 |
| `inst` | — | この試行の物体の個体（`sample_instance(seed, name)`。質量、摩擦、接触剛性、置き場所、向き、センサ雑音の種） |
| `set_palm(xyz)` | m | 掌の目標位置（ガントリは硬いサーボで追従） |
| `apply(bend)` | deg | 指 1 本の自由曲げ（筋肉の出力）。各関節の目標角はその 1/4 |
| `advance(n)` | — | 制御周期 n 回分を進める（物理は 1 kHz） |
| `obj_pos(name)` | m | 物体の中心 |
| `bend_true()` | deg × 3 | 各指の関節角の和 |
| `taxels_true()` | N × 24 | 指の内側の 24 区画に掛かる法線力（指・節・上下の順） |
| `force_true(name)` | N | 指と物体のあいだの法線力の和（評価用の真値。制御には使わない） |
| `plunger_set / plunger_pos / plunger_force` | m, m, N | 硬さプランジャ（MuJoCo のみ） |

## 2. 記号

| 記号 | 単位 | 意味 |
| --- | --- | --- |
| $p,\ \tilde p$ | kPa | 筋肉の内圧（真値）、内圧センサの値 |
| $p^{\mathrm{cmd}}$ | kPa | 内圧指令（50 Hz）。空圧供給は時定数 0.25 s の 1 次遅れ |
| $\theta^{\mathrm{free}},\ \hat\theta^{\mathrm{free}}$ | deg | 指 1 本の自由曲げ（何にも触れていないときの曲げ）、そのモデル予測 |
| $\bm\theta,\ \tilde{\bm\theta}$ | deg × 3 | 各指の実際の曲げ、曲げセンサの値 |
| $\bm s,\ \tilde{\bm s}$ | N × 24 | スキンの真値、センサの値 |
| $F,\ \hat F$ | N | 把持力（全指の法線力の和）の真値、固有感覚による推定 |
| $F^{*},\ \bar F$ | N | 目標把持力、つかみ直しの上限 |
| $\Lambda(s)$ | m | 付け根から $s$ の点に掛かる力に対する、全関節のてこの和 $\sum_j (s-s_j)^+$ |
| $k_b$ | N·m/rad | 関節 1 つの曲げ剛性（0.5） |
| $\kappa$ | — | 接触の判定（スキンの合計 > 0.10 N） |
| $k_p$ | N/mm | プランジャの荷重と押し込みの傾き |
| $\hat c$ | — | 取扱い区分（very_fragile、soft、fragile、sturdy） |

## 3. 1 回の把持

| 段階 | 時間 | 内容 |
| --- | --- | --- |
| approach / align | 1.3 s | 物体の上 22 cm へ。下りる直前にもう一度位置を読む |
| descend | 0.8 s | 握る高さへ（摩擦角の規則、`grasp_height`） |
| plunger | 約 4 s | 上面の 4 mm 上まで速く、そこから 2 mm/s で接触、1 mm/s で 0.3 N か 3 mm まで押す |
| grip | 最大 5 s | 接触まで 12 kPa/s、接触後は力の積分則。誤差 15 % 以内が 20 周期続いたら次へ |
| lift | 1.0 s | 20 cm 持ち上げる。物体が卓上に残れば「滑り」としてつかみ直す（最大 3 回） |
| transport | 2.0 s | トレーの上へ（最大加速度約 0.9 m/s²） |
| lower | 0.8 s | 置く高さへ。スキンの荷重が保持中より 0.6 N 増えたらそこで止める（置き損ねの防止） |
| release / retract | 1.4 s | 排気して離れる |

搬送中に推定値が目標の半分を 3 周期続けて下回ったら、物体を失ったとみなして手を開く。

## 4. ROS 2 のインタフェース（実機用、`ros2/softtwin_grasp_node.py`）

標準のメッセージ型だけを使うので、独自のインタフェースのビルドは要らない。

| 種類 | 名前 | 型 | 内容 |
| --- | --- | --- | --- |
| アクション | `/softtwin/grasp` | `control_msgs/action/GripperCommand` | `max_effort` = $F^*$ [N]。フィードバックは $\hat F$ と平均の曲げ |
| サービス | `/softtwin/zero` | `std_srvs/Trigger` | 手を開いた状態で曲げのゼロ点を取る |
| サービス | `/softtwin/release` | `std_srvs/Trigger` | 排気して把持を中止する |
| 購読 | `/real/pressure` | `std_msgs/Float32` | kPa（センサ用 QoS：best effort、depth 1） |
| 購読 | `/real/bend` | `std_msgs/Float32MultiArray` | deg × 3 |
| 購読 | `/real/skin` | `std_msgs/Float32MultiArray` | N × 24 |
| 配信 | `/softtwin/pressure_cmd` | `std_msgs/Float32` | kPa、50 Hz |

力のループは 50 Hz のタイマで回し、アクションの待ちは別のコールバックグループに置いた。遅いクライアントが力のループを止めないようにするためである。
このノードは rclpy の代わりの小さなスタブと MuJoCo のセルで CI 上の動作を確かめている（`tests/test_ros2_node.py`）。ROS 2 の実環境ではまだ動かしていない。

## 5. ファイル

| 場所 | 中身 |
| --- | --- |
| `sim_mujoco/` | 制御コア、MuJoCo のセル、触覚・プランジャの特徴、データ生成と Q2 の評価、PhysX の記録の再生、動画 |
| `sim/isaac_headless/` | PhysX のセル（USD）と、描画を使わない Kit アプリの設定 |
| `learning/` | 筋肉のモデル（MLP・グレーボックス）、区分の分類器、力の窓 |
| `scripts/` | 接触剛性の合わせ込み、判別マップ、頑健性、統計、図、本文の数値 |
| `results/` | 評価の生データ（1 試行 1 行）と `metrics.json` |
| `data/tactile_dataset/` | 自動アノテーション付きデータ（CC BY 4.0） |
| `core/`、`ros2/`、`firmware/` | 実機用の制御器、ROS 2 ノード、M5Stack のファームウェア（設計のみ） |
