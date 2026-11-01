# SoftTwin — 見た目が同じでも中身が違う食品を、力センサのない柔らかい指で壊さずにつかむ

[![ci](https://github.com/Qu0rise/softtwin/actions/workflows/ci.yml/badge.svg)](https://github.com/Qu0rise/softtwin/actions/workflows/ci.yml)

コード Apache-2.0 ／ データ CC BY 4.0 ／ Physical AI 講座 2026 応用編1 最終課題

![見た目は普通の中玉トマト、中身は過熟](media/lookalike_tomato.gif)

過熟の中玉トマトは、カメラには普通のトマトに見えます。見た目から「トマト＝2.2 N」で握ると潰れます（左）。
SoftTwin は把持の前に直径 8 mm のプランジャで 0.3 N だけ押して硬さを測り、柔らかい区分の 1.55 N で握ってトレーに置きます（右）。
指は空圧のシリコーン人工筋肉で、力センサはありません。把持力は、指の曲げ不足と、布状の触覚スキンが示す力の作用点から推定します。

| 結果 | |
| --- | --- |
| 力センサなしの閉ループ（5 物体 × 30 シード） | 成功 {{q1.learned.succ}} %、破損 0 / {{q1.n}}。直線の筋肉モデルでは {{q1.affine.succ}} %（いちごを全数破損） |
| 力の推定誤差（接触中） | 作用点をスキンで求めると {{q1.learned.mae}} N、指先と仮定すると {{q1.tip.mae}} N |
| 見た目が同じ 2 物体 | 見た目で決めると 0 %（全数破損）→ プランジャで決めると {{q2.plunger.la}} %、破損 {{q2.plunger.la.cr}}（正解の区分を与えた上限と同じ。目標の 85 % には未達） |
| 指で触るだけで見分けられるか | 過熟（硬さ 1/2.5）の判別は中玉トマト {{pm.tomato.finger}} %、いちご {{pm.strawberry.finger}} %。プランジャなら {{pm.tomato.plunger}} % |
| 物理エンジンを Isaac Sim（PhysX）に替える | 学習モデルの閉ループは PhysX でも {{px.learned.succ}} %・破損 0。直線モデルの成否は同じ個体で {{s2s.affine.same}} 一致（Cohen の κ_C = {{s2s.affine.kappa}}） |
| 物体の大きさが調整時と違う（1.2 倍） | 内圧表（開ループ）は過熟トマトを {{ss.recipe.1.2.tomato_overripe.cr}} 回潰す。閉ループは {{ss.learned.1.2.cr}} 回だが、重くなった物を落とす |

- 説明資料（PDF）：[`docs/paper/softtwin_report.pdf`](docs/paper/softtwin_report.pdf)　1 枚の要約：[`docs/paper/softtwin_summary.pdf`](docs/paper/softtwin_summary.pdf)
- 物理パラメータと文献：[`docs/physical_parameters.md`](docs/physical_parameters.md)　評価の数え方：[`docs/evaluation_protocol.md`](docs/evaluation_protocol.md)
- 数字の出どころ：[`results/metrics.json`](results/metrics.json)（このページの数字と説明資料の数字は、どちらもここから自動で書き出しています）

## 何をしたか

1. **一つの制御コア、二つの物理エンジン。** 物体の個体・筋肉・センサ・推定器・制御器・成否の判定を [`sim_mujoco/twin_core.py`](sim_mujoco/twin_core.py) にまとめ、MuJoCo と Isaac Sim 5.0 の PhysX はどちらも指と物体の動力学だけを受け持ちます。同じシードなら同じ個体（質量・摩擦・剛性・置き場所）を扱います。
2. **物理を文献値に合わせる。** 果実の接触剛性を N/mm で与え（トマトの硬さ 1.16〜3.58 N/mm、Vursavuş ら 2017 など）、MuJoCo では実効質量を測定から同定して目標の ±{{cal.mj.maxdev}} 以内に合わせました。PhysX でも同じ手順で測っています。
3. **作用点を触覚で求める固有感覚推定。** 指の曲げ不足 Δθ から力を出すには、力がどこに掛かるかが要ります：$\hat F_f = k_b\,\Delta\theta_f / \bar\Lambda_f$。$\Lambda(\xi)=\sum_j (\xi-\xi_j)^+$（$\xi$ は指に沿った位置） はてこの和で、$\bar\Lambda_f$ はスキン 8 点の出力で重み付けた $\Lambda$ の平均です。
4. **センサ構成を判別マップから決める。** 剛性比と雑音を掃引すると、想定した曲げセンサの雑音（0.3°）では、指で触るだけで文献の熟度差（比 1.3〜3.1）を見分けきれません。そこで掌に硬さプランジャを加えました。

![システム構成](docs/figures/fig_block_top.png)

外部入力は左端（物体、取扱い表）、出力は右端です。帰還は推定値の束 $\hat{\boldsymbol y}$ だけで、力センサの値は使いません。各ブロックの中身は [`docs/figures/fig_block_detail.png`](docs/figures/fig_block_detail.png)。

## Q1：力センサなしで潰さず落とさないか

![同じいちごで筋肉のモデルだけを替える](media/q1_strawberry.gif)

| 方式（MuJoCo、各 n = {{q1.n}}） | 成功 | 破損 | 推定誤差 |
| --- | --- | --- | --- |
| 内圧表（開ループ、区分ごとに調整） | {{q1.recipe.succ}} % | {{q1.recipe.cr}} | — |
| 閉ループ・直線の筋肉モデル | {{q1.affine.succ}} % | {{q1.affine.cr}} | {{q1.affine.mae}} N |
| 閉ループ・静特性の表 | {{q1.static.succ}} % | {{q1.static.cr}} | {{q1.static.mae}} N |
| 閉ループ・グレーボックス | {{q1.greybox.succ}} % | {{q1.greybox.cr}} | {{q1.greybox.mae}} N |
| 閉ループ・学習（提案） | {{q1.learned.succ}} % | {{q1.learned.cr}} | {{q1.learned.mae}} N |
| 同・作用点を指先に固定 | {{q1.tip.succ}} % | {{q1.tip.cr}} | {{q1.tip.mae}} N |
| 同・真の力で帰還 | {{q1.ideal.succ}} % | {{q1.ideal.cr}} | — |

直線モデルは把持に使う {{rec.lo}}〜{{rec.hi}} kPa で自由曲げを小さく見積もり、力も小さく見積もるので、いちごを全数潰しました。
不感帯と飽和を表せるモデル（表・グレーボックス・学習）はどれも破損 0 でした。差は豆腐の落下に出ます。把持中の推定値÷真値は、表 {{q1.static.bias}}、グレーボックス {{q1.greybox.bias}}、学習モデル {{q1.learned.bias}} です。

この偏りを、推定の入力を 1 つずつ真値に置き換えて分解しました（`scripts/estimator_error_split.py`、5 物体 × 10 シード）。

| 推定値÷真値 | 全体 | 筋肉モデル | 曲げセンサ | スキン | 作用点 |
| --- | --- | --- | --- | --- | --- |
| 静特性の表 | {{split.static.total}} | {{split.static.muscle}} | {{split.static.bend_sensor}} | {{split.static.skin}} | {{split.static.lever}} |
| グレーボックス | {{split.greybox.total}} | {{split.greybox.muscle}} | {{split.greybox.bend_sensor}} | {{split.greybox.skin}} | {{split.greybox.lever}} |
| グレーボックス＋遊び | {{split.greybox_play.total}} | {{split.greybox_play.muscle}} | {{split.greybox_play.bend_sensor}} | {{split.greybox_play.skin}} | {{split.greybox_play.lever}} |
| 学習モデル | {{split.learned.total}} | {{split.learned.muscle}} | {{split.learned.bend_sensor}} | {{split.learned.skin}} | {{split.learned.lever}} |

センサの雑音の寄与は 1 % 以下で、偏りは筋肉モデルと作用点の 2 項で決まります。作用点の項は約 +5 %（指先の丸みでの接触がスキンの最も先の点より先にあるため）。把持中の筋肉は履歴の上昇側の枝にいるので、履歴を持たないモデルは自由曲げを大きく見積もり、作用点の項と同じ向きに重なります。履歴（遊び要素）を加えたグレーボックスは、閉ループで {{q1.gbplay.succ}} % でした。ただし模擬筋肉そのものが同じ要素で作られているので、このモデルは構造を最初から知っています。学習モデルは構造を仮定せずに同じ水準に届きました。そのためには、同定データに接触後のゆっくりした増圧（1〜5 kPa/s）を入れる必要がありました。
区分が分かっていて、物体が調整に使ったものと同じ大きさなら、内圧表（開ループ）も閉ループと同じくらい強い基準です。エンジンを替えても、筋肉がずれても差は小さく出ました。閉ループの利点は、物体の大きさが調整時と違うときに破損を減らせることです（下の「大きさが変わったら」）。

![Q1 の結果](docs/figures/fig_q1.png)

## Q2：見た目が同じ物体を見分けられるか

![判別マップ](docs/figures/fig_probe_map.png)

指は果実より 16〜61 倍柔らかく（指先のコンプライアンス 0.041 m/N）、果実のへこみは指自身のたわみに埋もれます。
想定した曲げセンサの雑音（0.3°）では、剛性比 2.5 で見分けられるのは中玉トマト {{pm.tomato.finger}} %、いちご {{pm.strawberry.finger}} % です。
掌のプランジャは上面を 0.3 N だけ押して傾き $k_p$ を測ります。傾きには上面と台側の接触、手のたわみが直列に入りますが、手のたわみは個体によらないので判別を妨げません。想定の雑音では、剛性比 1.5 でトマト {{pm.tomato.plunger15}} %、比 2 以上で {{pm.tomato.plunger}} % です。

| 目標把持力の決め方（7 物体 × 30 シード） | 成功 | 破損 | 見た目が同じ 2 物体 | 1 回の時間 |
| --- | --- | --- | --- | --- |
| 見た目だけ | {{q2.prior.succ}} % | {{q2.prior.cr}} | {{q2.prior.la}} % | {{q2.prior.ct}} s |
| 弱く握り始め、滑ったら強める | {{q2.soft.succ}} % | {{q2.soft.cr}} | {{q2.soft.la}} % | {{q2.soft.ct}} s |
| 指プローブで決める | {{q2.finger.succ}} % | {{q2.finger.cr}} | {{q2.finger.la}} % | {{q2.finger.ct}} s |
| 硬さプランジャで決める（提案） | {{q2.plunger.succ}} % | {{q2.plunger.cr}} | {{q2.plunger.la}} % | {{q2.plunger.ct}} s |
| 同・作用点を指先に固定 | {{q2.tip.succ}} % | {{q2.tip.cr}} | {{q2.tip.la}} % | {{q2.tip.ct}} s |
| 同・重さの下限を加える | {{q2.mass.succ}} % | {{q2.mass.cr}} | {{q2.mass.la}} % | {{q2.mass.ct}} s |
| 力の窓を学習（区分を使わない） | {{q2.fw.succ}} % | {{q2.fw.cr}} | {{q2.fw.la}} % | {{q2.fw.ct}} s |
| 力の窓（その物体を学習に含めない） | {{q2.fwloo.succ}} % | {{q2.fwloo.cr}} | {{q2.fwloo.la}} % | {{q2.fwloo.ct}} s |
| 正解の区分を与える（判断の上限） | {{q2.oracle.succ}} % | {{q2.oracle.cr}} | {{q2.oracle.la}} % | {{q2.oracle.ct}} s |

プランジャで決めたときの失敗 {{fail.plunger.n}} 件は、正解の区分を与えた上限の失敗と同じ試行で、すべて搬送中の落下でした。判断はもう誤っておらず、残りは握り方の問題です（重い過熟トマト、指のあいだから抜ける過熟いちご、滑りやすい豆腐）。作用点を指先に固定すると力を 1 割強めに握るので落下が減り {{q2.tip.la}} % になりますが、過熟いちごを {{q2.tip.la.cr}} 回潰しました。推定は正しいまま、見た目の種類の名目の質量と摩擦から目標の下限を上げると（下の「大きさが変わったら」）、過熟トマトと豆腐の落下が減って {{q2.mass.la}} %（破損 0）になります。残る過熟いちごは力を上げる余地がなく、握り方を変える必要があります。

![Q2 の結果](docs/figures/fig_q2.png)

## Q3：生成データだけで判断を学習できるか

![自動アノテーション付きデータの例](docs/figures/dataset_sample.png)

{{ds.n}} 試行（{{ds.samples}} サンプル）を 2 CPU で {{ds.min}} 分で生成し、全試行が検証器を通りました。区分の正解率（5 分割交差検証）は、画像のみ {{cv.vision}} %、画像＋指プローブ {{cv.finger}} %、画像＋プランジャ {{cv.plunger}} % です。
LeRobotDataset v2.1 の構成にも書き出せます（`scripts/export_lerobot.py`）。

## 物理エンジンを替える（MuJoCo → Isaac Sim 5.0 / PhysX）

![同じ制御コアを 2 つのエンジンで](media/sim2sim_tomato.gif)

制御コアは同じまま、指と物体の動力学だけを PhysX（コンプライアント接触、TGS、1 kHz）に替えました。
同じシードは同じ個体なので、試行ごとに突き合わせられます。学習モデルの閉ループは PhysX で {{s2s.learned.px}} %（同じ個体の MuJoCo で {{s2s.learned.mj}} %）、破損はどちらも 0 です。両エンジンとも成功がほぼ全数なので、一致数そのものは目安にとどまります。失敗の多い直線モデルでは成否の一致が {{s2s.affine.same}}（Cohen の κ_C = {{s2s.affine.kappa}}）で、いちごは PhysX でも全数潰れました。不一致 {{s2s.diff.n}} 件のうち {{s2s.diff.tofu}} 件は、MuJoCo で落ちた豆腐を PhysX では運べたものです（PhysX ではどの方式でも豆腐を落としませんでした）。
柔らかい 3 物体では同じ個体のピーク力の差は中央値 {{s2s.learned.rel.soft}} %、硬い卵とボルトでは {{s2s.learned.rel.stiff}} % で、差は搬送の加減速の瞬間にだけ出ます。
実行環境の GPU ドライバが Isaac Sim 5.0 の RTX 描画に対応していないため、PhysX は描画なしで計算しました。動画の右は、PhysX が計算した姿勢を MuJoCo の描画器で表示したものです。

## 筋肉と指が想定とずれたら

![想定とのずれ](docs/figures/fig_robustness.png)

筋肉の 6 つの定数を ±30 % ずらすと、学習モデルの成功率は {{rob.learned.30}} % に下がります。空中較正で実測とモデルの差を足すと {{rob.learnedcal.30}} % まで回復しました（同じ条件の内圧表は {{rob.recipe.30}} %）。
指の曲げ剛性 $k_b$ のずれは力の推定にそのまま比例して入り、空中較正では見えません（25 % 硬いとピーク÷目標が {{rob.kb100.pk}} から {{rob.kb125.pk}} へ）。実機ではロードセルを一度押す校正が要ります。

## 物体の大きさが変わったら

内圧表（開ループ）は、調整に使った大きさの物体でしか力を保証しません。脆い 3 物体と過熟トマトの寸法を 0.8 倍・1.2 倍にしました（質量は 3 乗、両方式に正しい区分を与える、各 n = 80）。

| 倍率 | 内圧表：成功 | 破損 | ピーク÷目標 | 閉ループ：成功 | 破損 | ピーク÷目標 | ＋重さの下限：成功 | 破損 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.8 | {{ss.recipe.0.8.succ}} | {{ss.recipe.0.8.cr}} | {{ss.recipe.0.8.pk}} | {{ss.learned.0.8.succ}} | {{ss.learned.0.8.cr}} | {{ss.learned.0.8.pk}} | {{ss.mass.0.8.succ}} | {{ss.mass.0.8.cr}} |
| 1.0 | {{ss.recipe.1.succ}} | {{ss.recipe.1.cr}} | {{ss.recipe.1.pk}} | {{ss.learned.1.succ}} | {{ss.learned.1.cr}} | {{ss.learned.1.pk}} | {{ss.mass.1.succ}} | {{ss.mass.1.cr}} |
| 1.2 | {{ss.recipe.1.2.succ}} | {{ss.recipe.1.2.cr}} | {{ss.recipe.1.2.pk}} | {{ss.learned.1.2.succ}} | {{ss.learned.1.2.cr}} | {{ss.learned.1.2.pk}} | {{ss.mass.1.2.succ}} | {{ss.mass.1.2.cr}} |

閉ループは大きさによらず目標の約 1.1 倍で握り、1.2 倍でも潰しません。ただし大きく重い豆腐と過熟トマトを搬送中に落とし、成功率は内圧表を下回りました。内圧表は大きい物ほど強く握るので落下は減りますが、許容力の幅が狭い過熟トマトでは、その強さが破損になります。

物体が大きくなって変わるのは、潰さない力の上限ではなく落とさない力の下限です。そこで、カメラで測った大きさから推した重さで目標の下限を上げました（$F^* = \min(\max(H_{\rm t}, S\hat m(g+a)/\hat\mu), {{ss.mass.cap}}H_{\rm lim})$、$S$ は等倍の物体だけから決めた {{ss.mass.safety}}）。等倍の過熟トマトは目標が 1.55 → {{ss.mass.F.tomato_overripe.1}} N になって落下が減りますが、1.2 倍の過熟トマトに要る {{ss.mass.Fmin.tomato_overripe.1.2}} N は許容力 2.0 N を超え、潰さず落とさない力の幅がありません。これは握る前に計算で分かるので、実際の工程では見送って別の把持に回すのがよいと考えます。

## 開発に使っていないシードでの確認

取扱い表と評価物体は開発中に評価シード 0〜29 の結果を見て見直したので、最終の構成を固定して、一度も使っていないシード {{ho.seeds}} で Q1 と Q2 を再実行しました。

| | シード 0〜29 | シード {{ho.seeds}} |
| --- | --- | --- |
| Q1 閉ループ・学習（成功・破損） | {{q1.learned.succ}} %・{{q1.learned.cr}} | {{ho.q1.learned.succ}} %・{{ho.q1.learned.cr}} |
| Q1 閉ループ・直線（成功・破損） | {{q1.affine.succ}} %・{{q1.affine.cr}} | {{ho.q1.affine.succ}} %・{{ho.q1.affine.cr}} |
| Q2 見た目だけ（見た目が同じ 2 物体） | {{q2.prior.la}} %・破損 {{q2.prior.la.cr}} | {{ho.q2.prior.la}} %・破損 {{ho.q2.prior.la.cr}} |
| Q2 プランジャで決める（提案） | {{q2.plunger.la}} %・破損 {{q2.plunger.la.cr}} | {{ho.q2.plunger.la}} %・破損 {{ho.q2.plunger.la.cr}} |
| Q2 同・重さの下限を加える | {{q2.mass.la}} %・破損 {{q2.mass.la.cr}} | {{ho.q2.mass.la}} %・破損 {{ho.q2.mass.la.cr}}（区間 {{ho.q2.mass.la.ci}}） |

重さの下限の係数はシード 0〜19 だけで決めたので、右の列はその独立な検証です。

## 制約

- 評価はすべてシミュレーションです。人工筋肉とセンサは仮定した特性のモデルで、実機は未製作です。
- 果実は剛体の外形と接触の柔らかさで近似し、局所的な変形と損傷の進行は扱っていません。許容力は卵以外は仮定値です。
- 過熟の剛性比（2.5）は仮定で、判別マップで比への依存を示しました。物体の位置はシミュレーションの真値です。
- PhysX で評価したのは Q1 だけです。
- Q1・Q2 の目標力は区分だけで決めました。重さの下限（大きさの節）は名目の質量と摩擦を使い、個体の重さは見ていません。

## 再現する

```text
pip install -r requirements.txt
unzip -n data/tactile_dataset.zip -d data/
python scripts/analysis_stats.py && python scripts/make_numbers.py      # 生データ -> metrics.json -> 本文と README の数字
python sim_mujoco/softtwin_mj.py bench --seeds 2 --out /tmp/q1.json     # Q1 をシード 0-1 で再実行（約 1 分）
python scripts/check_determinism.py /tmp/q1.json                        # 公開した行と一致するか
pytest -q
```

すべての実験を最初から回す手順と所要時間は [`docs/reproducibility.md`](docs/reproducibility.md) にあります（2 CPU で約 1 時間、PhysX は別に約 3 時間）。

## 構成

| 場所 | 内容 |
| --- | --- |
| `sim_mujoco/` | 制御コア `twin_core.py`、MuJoCo のセル、触覚とプランジャの特徴、データ生成と Q2、PhysX の記録の再生、動画 |
| `sim/isaac_headless/` | Isaac Sim 5.0 / PhysX のセル（USD）と、描画を使わない Kit アプリ |
| `learning/` | 筋肉のモデル（MLP・グレーボックス）、区分の分類器、力の窓 |
| `scripts/` | 接触剛性の合わせ込み、判別マップ、頑健性、統計、図、本文の数字 |
| `results/` | すべての数字の元データ（1 試行 1 行）と `metrics.json` |
| `core/`、`ros2/`、`firmware/` | 実機用：同じ制御コアを呼ぶ制御器、ROS 2 ノード（アクション・サービス）、M5Stack（設計のみ） |
| `docs/` | 物理パラメータ、評価の数え方、設計判断、モデルカード、実機への移行、説明資料、図 |

## English summary

SoftTwin grasps look-alike soft foods (an overripe mid-size tomato that looks like a ripe one) with a soft pneumatic gripper that has no force sensor.
One control core (object instances, muscle, sensor models, estimator, controller, outcome labels) runs unchanged on MuJoCo and on Isaac Sim 5.0 / PhysX; contact stiffness is set in N/mm from the literature and identified in both engines.
The grip force is estimated from the bending deficit caused by the contact, divided by the lever-arm sum weighted with the fabric skin readings (contact MAE {{q1.learned.mae}} N);
closed-loop grasping succeeds in {{q1.learned.succ}} % with 0/{{q1.n}} crushed (linear muscle model: {{q1.affine.succ}} %, all strawberries crushed).
A discrimination map shows that squeezing with the soft fingers cannot reliably tell ripe from overripe at realistic stiffness ratios ({{pm.tomato.finger}} % for tomatoes at ratio 2.5), so an 8 mm flat plunger probes the top first:
look-alike success rises from 0 % (all crushed) to {{q2.plunger.la}} % with no crush; the remaining failures are drops on exactly the trials that also fail with the true class given (target 85 %, not met).
A per-class pressure recipe is as good when the objects match the ones it was tuned on, but at 1.2x size it crushes the overripe tomato {{ss.recipe.1.2.tomato_overripe.cr}} times (closed loop: {{ss.learned.1.2.cr}}, but it drops the heavier objects; a weight-based force floor helps until the floor exceeds the allowable force).
All evaluation is in simulation; the muscle and the sensors are assumed models.

## ライセンスと引用

コードは Apache-2.0、データ（`data/`）は CC BY 4.0 です。引用は [`CITATION.cff`](CITATION.cff) を使ってください。
