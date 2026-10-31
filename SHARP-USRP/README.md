# SHARP-USRP

USRP B205mini-i で取得・デコードした Wi-Fi CSI を、SHARP の位相サニタイゼーション
アルゴリズムに入力するためのシステム。

対象とする実験条件:

| | 規格 | 帯域 | チャネル | アンテナ | 設定名 |
|---|---|---|---|---|---|
| パターン1 | Wi-Fi 5 (802.11ac, VHT) | 20 MHz | Ch36 | 1 (SISO) | `VHT20` |
| パターン2 | Wi-Fi 6 (802.11ax, HE) | 20 MHz | Ch36 | 1 (SISO) | `HE20` |

## 結論: 変換は可能

SHARP の位相サニタイゼーションが必要とするのは **複素チャネル周波数応答 (CFR) の
サブキャリア方向の並び** だけで、それがどのハードウェアで得られたかには依存しない。
Nexmon CSI (pcap 経由) でも USRP + MATLAB WLAN Toolbox のチャネル推定でも、
「パケットごとの複素 CFR ベクトル」という点では同じものである。

必要なのは次の 2 つだけ:

1. **形式の変換** — USRP デコーダは使用サブキャリアのみ (VHT20 なら 56 本) を
   出力するが、SHARP は FFT 長ぶんの配列 (ヌル込み) を前提とする。
2. **帯域のパラメータ化** — SHARP / SHARPax はどちらも 80 MHz 決め打ちで、
   FFT 長・ヌルサブキャリア番号・切り出し範囲がハードコードされている。
   20 MHz 用に一般化する必要がある。

本リポジトリはこの 2 点を実装したもの。実データでの検証結果は「[検証](#検証)」を参照。

### なぜ USRP のデータで位相サニタイゼーションが特に必要か

USRP は AP と位相同期していないため、パケットごとに**ランダムな共通位相**と
**時刻オフセット** (パケット検出位置のずれ = 周波数方向の位相傾斜) が乗る。
実測データでは生の CSI の位相はパケット間でほぼランダム (標準偏差 1.96 rad)
であり、そのままでは位相を使った解析ができない。

SHARP の手法は、CFR を「遅延の異なる複数経路の和」としてスパース推定し、
**最強経路の位相を基準に全経路を再合成する**ことでこれらのオフセットを相殺する。
これはまさに USRP のデータが必要としている処理である。

## データの流れ

```
USRP B205mini-i
   │  captureIQ.m
   ▼
生IQ (.mat)
   │  decodeIQ_VHT.m  (Wi-Fi 5)  /  decodeIQ_HE.m (Wi-Fi 6)
   ▼
デコード済み CSI (.mat)              csiVHT [パケット x 56] など
   │
   │  ★ 手順0: usrp_to_sharp.py  または  matlab/usrpCSItoSHARP.m
   ▼
SHARP 入力形式 (.mat)                csi_buff [パケット x FFT長] (FFT順)
   │
   │  ★ 手順1: CSI_phase_sanitization_signal_preprocessing.py
   ▼
signal_<name>.txt                    [使用サブキャリア x パケット x ストリーム]
   │
   │  ★ 手順2: CSI_phase_sanitization_H_estimation.py
   ▼
Tr_vector_<name>_stream_N.txt        位相基準化済みの CFR
   │
   │  ★ 手順3: CSI_phase_sanitization_signal_reconstruction.py
   ▼
processed_phase/<name>_stream_N.mat  csi_matrix_processed
                                     [パケット x サブキャリア x (振幅,位相)]
                                     time_sec (パケット取得時刻)
   │
   │  ★ 手順4: CSI_doppler_computation.py
   ▼
doppler_traces/<name>.txt            [時間窓 x ドップラービン]
   │
   │  ★ 手順5: CSI_doppler_plot.py
   ▼
plots/<name>.png                     ドップラーマップ
```

`csi_matrix_processed` は SHARP オリジナルの出力と同じ変数名・同じ意味を持つ。
手順3 は加えて `time_sec` (パケット取得時刻) も保存する。これは後段で
実測のサンプリング間隔を使うために必要で、オリジナルには無い。

## 準備

```bash
pip install numpy scipy osqp h5py
```

`h5py` は MATLAB v7.3 (HDF5) 形式の .mat を読むために必要。デコーダは `-v7.3` で
保存するため実質必須。

## 使い方

### 一番簡単な方法 (手順0〜3を一括実行)

```bash
cd Python_code
python3 run_phase_sanitization.py ../../CSI/202608191054_OpenWrt-A_CSI.mat
```

規格 (VHT / HE) はファイルの中身から自動判定される。結果は
`processed_phase/<入力名>_stream_0.mat` に出る。

活動ラベルごとにサブディレクトリを分けたい場合:

```bash
python3 run_phase_sanitization.py ../../CSI/xxx.mat --name walk01 --label W
# -> processed_phase/W/walk01_stream_0.mat
```

主なオプション:

| オプション | 意味 |
|---|---|
| `--format VHT` / `--format HE` | 使う PHY 形式を明示 (既定はパケット数が最多のもの) |
| `--fcs_only` | FCS 検証済みパケットのみ使う |
| `--start_idx` / `--end_idx` | 先頭・末尾から捨てるパケット数 |
| `--subcarriers_space` | サブキャリアの間引き幅 (既定は規格ごとの推奨値) |
| `--delta_t_refined` | 精細グリッドの時間刻み [s] |

### 段階ごとに実行する方法

SHARP オリジナルと同じ引数体系を保っている (末尾に本実装固有のオプションを追加)。

```bash
cd Python_code

# 手順0: SHARP 形式へ変換
python3 usrp_to_sharp.py ../../CSI/xxx_CSI.mat ./input_files/exp01.mat

# 手順1: 前処理  <dir> <all_dir> <name> <nss> <ncore> <start_idx>
python3 CSI_phase_sanitization_signal_preprocessing.py ./input_files/ 0 exp01 1 1 0

# 手順2: 多重波推定と位相の基準化  ... <start_r> <end_r>
python3 CSI_phase_sanitization_H_estimation.py ./input_files/ 0 exp01 1 1 0 -1

# 手順3: 再構成  <dir> <dir_save> <nss> <ncore> <start_idx> <end_idx>
python3 CSI_phase_sanitization_signal_reconstruction.py ./phase_processing/ ./processed_phase/ 1 1 0 0
```

SISO・アンテナ1本なので `nss=1`, `ncore=1` を指定する
(オリジナルの SHARP は Nexmon の 4 コア構成で `ncore=4` だった)。

### 複数ファイルを一括で処理する (手順0〜5)

夜間に流しっぱなしにする用途。ワイルドカードで複数指定できる。

```powershell
cd SHARP-USRP\Python_code
python run_batch.py "D:\IQ_csi\0927*_WAX202_CSI.mat"
```

これだけで、各ファイルについて 変換 → 前処理 → 多重波推定 → 再構成 →
ドップラー計算 → 描画 まで通り、`plots/` に PNG が出る。

夜間実行を想定した作り:

| | |
|---|---|
| 1 件が失敗しても | 残りを処理し続け、最後に成否の一覧を出す |
| 画面を閉じても | `batch_<日時>.log` に経過が残る |
| 再実行すると | PNG がある分は飛ばす (`--redo` でやり直し) |

所要時間の目安は **4956 パケットで約 5 分**（ほぼ全て手順2 の多重波推定）。
パケット数にほぼ比例する。

主なオプションは `run_phase_sanitization.py` と `CSI_doppler_computation.py`
のものを引き継いでいる (`--label`, `--resample`, `--fc`, `--sample_length` など)。

### ドップラーマップを出す (手順4〜5)

```bash
# 手順4: ドップラースペクトルの計算
#   <dir> <subdirs> <dir_doppler> <start> <end> <窓長> <ずらし幅> <雑音床>
python3 CSI_doppler_computation.py ./processed_phase/ . ./doppler_traces/ 10 10 31 1 -1.5

# 手順5: 画像として描画
python3 CSI_doppler_plot.py ./doppler_traces/ ./plots/
```

`--label` でサブディレクトリを分けている場合は `.` の代わりに
`W,R,S,E` のようにカンマ区切りで指定する。

**動きが見えないときは --remove_static を試す**

直接波や家具などの静止経路は動かないので、CFR の時間平均に集まる。人の動きによる
反射はそれより遥かに弱く、実測 (Wi-Fi 5, 歩行) では **静止成分より 11.4 dB 小さい**。
SHARP と同じ既定 (`--noise_level -1.5` = -15 dB で切り捨て) では、**スペクトルの
ビンの 89.6% が雑音床に潰れてしまう**。

`--remove_static` は窓内の時間平均を各サブキャリアから引いてから変換する。
これで動き成分と静止成分の比が -11.4 dB → +3.3 dB (15 dB 改善) になる。
30 秒間動き続けた測定が数回の短い突発にしか見えない場合、まずこれを疑うこと。

```bash
python3 CSI_doppler_computation.py ./processed_phase/ . ./doppler_traces/ 10 10 31 1 -1.5 --remove_static
```

注意: 0 m/s の静止成分は消える。SHARP の学習側に入れる場合は、既定 (除去しない)
と揃えるか、学習データ全体で同じ設定にすること。

**全速度域を貫く縦縞が出たら --despike**

ドップラーマップに、全速度域 (±5 m/s 全体) を一本で貫く縦縞が現れることがある。
人の動きは数 m/s までなので、全速度域に同時にエネルギーが出るのは物理的でない。

原因は **1 パケットの外れ値** (振幅の飛び、位相のずれ)。検証したところ、外れ値が
1 つあるだけで STFT の窓長ぶんの窓すべてに全速度域へ広がるエネルギーが乗り、
縦縞 1 本として現れる。

`--despike` は前後どちらとの差も大きい孤立した飛びを検出し、線形補間で置き換える。
実測では縦縞の指標 (端へのパワー漏れ) が 0.317 → 0.080 に下がった。

```bash
python3 CSI_doppler_computation.py ./processed_phase/ . ./doppler_traces/ 10 10 31 1 -1.5 --remove_static --despike
```

閾値 `--despike_sigmas` の既定は 8 (実測データで 0.5% のパケットが該当)。
下げると多く除去できるが速い動きも消しかねない (5 にすると 2.1%)。

なお SHARP の前処理にも `hampel_filter` が定義されているが呼ばれておらず、
オリジナルではこの対処が有効になっていない。

**取得レートが違う測定は --sub_sampling でそろえる**

速度軸の範囲は `v_max = c / (Tc · fc) / 2`、速度分解能は `c / (Tc · fc · 窓長)` で
決まる。どちらも Tc に依存するため、**取得レートが違う測定どうしは軸の範囲も
分解能も揃わない**。

| | Tc | 速度軸 | 窓の実時間 | 速度分解能 |
|---|---|---|---|---|
| 170 pkt/s, 窓 31 | 5.79 ms | ±5.0 m/s | 179 ms | 0.32 m/s |
| 860 pkt/s, 窓 31 | 1.16 ms | **±25 m/s** | 36 ms | **1.61 m/s** |
| 860 pkt/s, 窓 31, **間引き 1/5** | 5.79 ms | ±5.0 m/s | 179 ms | 0.32 m/s |
| 860 pkt/s, 窓 155 | 1.16 ms | ±25 m/s | 180 ms | 0.32 m/s |

分解能が歩行速度 (約 1 m/s) を超えると、動きが静止成分の主ローブに埋もれて
分離できない。ドップラーマップ上では**静止成分の帯が異常に太く**見え、
**速度軸の範囲も広くなる**ので判別できる。

`--sub_sampling N` は N 個に 1 個へ間引いて Tc を N 倍にする。上の表のとおり
**軸の範囲と分解能の両方が揃う**ので、レートの違う測定を比べるならこれを使う
(SHARPax の同名オプションと同じ考え方)。

```bash
# 860 pkt/s の測定を 170 pkt/s の測定に揃える
python3 CSI_doppler_computation.py ./processed_phase/ . ./doppler_traces/ 10 10 31 1 -1.5 --remove_static --sub_sampling 5
```

窓長を伸ばす (`--sample_length 155`) 方法でも分解能は揃うが、軸の範囲は広いままに
なる。間引くと `v_max` が下がるので、それを超える速度は折り返す点に注意
(人の動きは 3 m/s 程度までなので通常は問題にならない)。

図の見た目だけ揃えたい場合は描画側の `--ylim` を使う。

```bash
python3 CSI_doppler_plot.py ./doppler_traces/ ./plots/ --ylim 5
```

**重要 — SHARP のドップラー計算をそのまま使ってはいけない理由**

SHARP の `CSI_doppler_computation.py` はアルゴリズム自体 (ハン窓 → FFT →
パワー → サブキャリア方向に加算 → 正規化) はそのまま使えるが、
コードをそのまま流用すると次の 3 点で誤る。

| # | 問題 | 症状 |
|---|---|---|
| 1 | `--bandwidth 20` の意味が逆転する | **エラーが出ずに誤った結果**になる |
| 2 | `Tc = 6e-3`, `fc = 5e9` のハードコード | 速度軸の目盛りがずれる |
| 3 | 非等間隔サンプリング | 速い動きでスペクトルがにじむ |

**① `--bandwidth 20` の罠**

SHARP の `--bandwidth 20` は「80MHz で取得したデータから 20MHz 幅を切り出す」
オプションで、245 列 (80MHz 分) の配列に対する固定インデックス
(`np.arange(0, 57)` など) である。本システムの入力はすでに 20MHz なので:

- **VHT20** (57列): `--sub_band 1` はたまたま全域と一致して動くが偶然。
  `--sub_band 2` 以降は範囲外エラー
- **HE20** (245列): 20MHz のさらに 1/4 だけを切り出してしまう。
  列数が SHARP の 80MHz と同じ 245 なので**エラーが出ず気付けない**

本実装ではこのオプションを廃止し、既定で全サブキャリアを使う。部分帯域を
見たい場合は `--subcarrier_range LO HI` で明示する。

**② Tc と fc**

本実装は手順3 が保存した `time_sec` から Tc を実測する。中心周波数は
`--fc` で指定する (既定 5.18e9 = Ch36)。速度軸の目盛りは
`_meta.json` に記録され、描画時に使われる。

なお本実装ではビン間隔を FFT のビン数 `n_fft` から計算する
(`Δv = c / (Tc · fc · n_fft)`)。窓長から決まるのは分解能
(`c / (Tc · fc · 窓長)`) で、ゼロ詰めしても向上しない量なので別に表示している。

**③ 非等間隔サンプリング — 実測では問題にならない**

これは事前に懸念していたが、実測の時刻列 (変動係数 0.15) で検証したところ
**そのまま等間隔とみなすのが最も素直で、補間はむしろ有害**だった。

真の速度 ±4 ビンへのパワー集中度 (高いほどにじみが少ない):

| ドップラー速度 | 理想(等間隔) | `none` (既定) | `nudft` | `interp` (線形補間) |
|---|---|---|---|---|
| 0.5 m/s | 0.595 | **0.592** | 0.547 | 0.592 |
| 1.0 m/s | 0.330 | **0.323** | 0.303 | 0.309 |
| 2.0 m/s | 0.330 | 0.302 | 0.302 | 0.247 |
| 3.0 m/s | 0.330 | 0.277 | **0.306** | 0.165 |

- **2 m/s 程度まで (人の歩行) は既定の `none` で理想とほぼ同等**
- 3 m/s 付近では `none` が理想の 84% まで落ち、`nudft` (非等間隔DFT) が取り戻す。
  走行など速い動きを見るなら `--resample nudft`
- **線形補間による等間隔化は高速側で明確に悪化する** (補間が低域通過フィルタ
  として働くため)。比較目的以外では使わないこと

実データでも `none` と `nudft` の結果は相関 0.9996 でほぼ一致した。
検証コードは `tests/test_doppler.py`。

なお、これは「AP 側から一定レートでトラフィックを流す」必要が無いという意味
ではない。間隔のばらつきが今回より大きい測定では `nudft` を使うか、測定側で
レートを安定させるほうがよい。

### 活動認識の学習とテスト (手順6〜8)

SHARP の学習側 (データセット作成・ネットワーク・評価) を USRP の測定に合わせて
移植したもの。ここでは S1 で学習し、S2・S3 でテストする例を示す。

```
doppler_traces/<S>/*.txt            ドップラーデータ (手順4 の出力)
   │  ★ 手順6: CSI_doppler_create_dataset_train.py (学習用) / _test.py (テスト用)
   ▼
doppler_traces/<S>/train_... など    窓に切ったデータセット [アンテナ x ドップラービン x 時間]
   │  ★ 手順7: CSI_network.py (学習)
   ▼
usrp_<アンテナ>_<活動>_network.keras 学習したモデル
   │  ★ 手順8: CSI_network_test.py (テスト)
   ▼
outputs/, plots/                     正解率・F 値・混同行列
```

**準備**

```bash
pip install tensorflow scikit-learn matplotlib
```

Windows でネイティブに入れた TensorFlow は CPU でしか動かないが、ネットワークが
小さいので CPU で十分学習できる。

**1 コマンドで全部回す (run_har.py)**

下の手順6〜8 を順に実行し、最後に正解率の表と混同行列をまとめて出す。
活動ラベルはファイル名から集めるので指定しなくてよい。

```powershell
cd SHARP-USRP\Python_code

# ドップラーデータ (doppler_traces/) ができている場合
python run_har.py --train S1 --test S2,S3

# デコード済み CSI (E:\Dataset_CSI\S1\W_antenna1.mat など) から全部
python run_har.py --train S1 --test S2,S3 --csi E:\Dataset_CSI --noise_level -3.0 --despike

# アンテナ 2 本と 1 本の両方で回して比べる
python run_har.py --train S1 --test S2,S3 --antennas antenna1,antenna2 --antennas antenna1
```

| | |
|---|---|
| `outputs/har_summary_<日時>.md` | 正解率・F 値の表と混同行列 (画面にも出る) |
| `plots/complete_different_*.png` | 混同行列の図 |
| `har_<日時>.log` | 全工程の画面出力 |

- `--csi` を付けると、その前に `run_batch.py` でドップラーデータを作る
  (`<csi>/S1/*.mat` などを処理。完了済みのファイルは飛ばす)。run_har.py が知らない
  オプション (`--noise_level`, `--despike` など) は run_batch.py にそのまま渡す。
  位相サニタイゼーションに **約 5 分 / 5000 パケット** 掛かるので、CSI から始めると
  全体で数時間になる
- テストのセッションが複数あると、まとめた評価に加えてセッションごとにも評価する
- どこかの工程が失敗したらその時点で止め、失敗した工程を表示する

**ファイルの置き方**

ドップラーデータは次のどちらの置き方でもよい。データセットはどちらの場合も
`doppler_traces/S1/` などのサブディレクトリに書き出す。

```
Python_code/doppler_traces/                  Python_code/doppler_traces/
├── S1_E_antenna1_stream_0.txt               ├── S1/
├── S1_E_antenna1_stream_0_meta.json         │   ├── S1_E_antenna1_..._stream_0.txt
├── S1_E_antenna2_stream_0.txt               │   ├── S1_E_antenna2_..._stream_0.txt
├── S1_W_antenna1_stream_0.txt               │   └── ...
├── ...                                      ├── S2/
└── S3_W_antenna2_stream_0.txt               └── S3/
```

- ファイル名はセッション名 (`S1`) で始まり、**4 文字目** (`S1_E_...` の `E`) が
  活動ラベル (SHARP は `S1a_E_...` の 5 文字目)
- `antenna1` / `antenna2` がアンテナ。アンテナ名以外が同じファイルどうしを
  同じ測定として束ねる
- `_meta.json`, `_time.npy` はそのまま置いておいてよい (読み込み対象から外している)
- 2 台の受信機のデータ (`W_antenna1.mat` と `W_antenna2.mat` など) は、`alignCSI.m`
  の出力 (`*_aligned_CSI.mat`) を使う。2 台が両方受けたパケットだけが同じ順番・
  同じ時刻で入っているので、パケット番号で切るドップラーの窓も 2 台で同じパケットを
  指す。元の `*_CSI.mat` は受けたパケットが 2 台で少しずつ違い、番号で重ねると
  後ろへ行くほど時刻がずれる

左の形は、セッションごとのフォルダ (`E:\Dataset_CSI\S1\W_antenna1.mat` など) を
`run_batch.py` にまとめて渡すとできる (`--prefix_parent` で出力名の頭に必ず
セッション名を付ける)。右の形は `--label S1` を付けてセッションごとに処理したもの。
ドップラー計算の設定 (`--noise_level`, `--remove_static` など) は S1〜S3 で揃えること。

```powershell
python run_batch.py "E:\Dataset_CSI\S1\*.mat" "E:\Dataset_CSI\S2\*.mat" "E:\Dataset_CSI\S3\*.mat" --prefix_parent --noise_level -3.0 --despike
```

**段階ごとに実行する**

活動ラベル `E,L,W,R,J` は例。実際に使うラベルに置き換える。

```powershell
cd SHARP-USRP\Python_code

# 手順6: データセット作成
#   S1 は測定ごとに時間方向で 学習 60% / 検証 20% / テスト 20% に分ける
#   S2, S3 は分けずに全区間をテストに使う
python CSI_doppler_create_dataset_train.py ./doppler_traces/ S1 31 1 340 30 E,L,W,R,J antenna1,antenna2
python CSI_doppler_create_dataset_test.py ./doppler_traces/ S2,S3 31 1 340 30 E,L,W,R,J antenna1,antenna2

# 手順7: 学習 (S1)。最後に S1 から取り分けたテスト区間での結果を表示する
python CSI_network.py ./doppler_traces/ S1 100 340 1 32 antenna1,antenna2 usrp E,L,W,R,J

# 手順8: テスト (S2,S3) と混同行列の描画
python CSI_network_test.py ./doppler_traces/ S2,S3 100 340 1 32 antenna1,antenna2 usrp E,L,W,R,J
python CSI_network_metrics_plot.py complete_different_antenna1,antenna2_E,L,W,R,J_S2,S3 E,L,W,R,J
```

| 引数 | 意味 |
|---|---|
| `31 1` | 手順4 の窓長とずらし幅。学習・検証・テストの境目で窓が重ならないよう空ける幅に使う |
| `340 30` | 学習の 1 サンプルの時間方向の長さ (ドップラーの時間窓 340 個) とずらし幅。170 pkt/s なら約 2 秒 |
| `100` | ドップラービン数 (= 手順4 の `--n_fft`) |
| `1 32` | チャネル数、バッチサイズ |
| `antenna1,antenna2` | 使うアンテナ |
| `usrp` | モデルなどのファイル名の先頭 (任意) |

`S2,S3` のように並べるとまとめて 1 つの評価になる。別々に見たいときは
`S2` と `S3` で 1 回ずつ実行する。学習の最大エポック数は `--epochs` で変えられる
(既定 25。検証損失が 3 エポック改善しなければ早めに止まる)。

**アンテナを 1 本にする**

アンテナの指定を `antenna1` に変えるだけでよい。データセット・モデル・結果の
ファイル名に使ったアンテナが入るので、2 本のときの結果は上書きされない。
`2` のように本数で書くと `antenna1,antenna2` と同じ意味になる。

```powershell
python CSI_doppler_create_dataset_train.py ./doppler_traces/ S1 31 1 340 30 E,L,W,R,J antenna1
python CSI_doppler_create_dataset_test.py ./doppler_traces/ S2,S3 31 1 340 30 E,L,W,R,J antenna1
python CSI_network.py ./doppler_traces/ S1 100 340 1 32 antenna1 usrp E,L,W,R,J
python CSI_network_test.py ./doppler_traces/ S2,S3 100 340 1 32 antenna1 usrp E,L,W,R,J
```

ネットワークはアンテナ 1 本ぶんの入力を受け取り、複数のアンテナは評価のときに
予測を統合する (SHARP と同じ)。そのため、2 本で学習したモデルを antenna1 だけで
評価することもできる (`--model usrp_antenna1,antenna2_E,L,W,R,J_network.keras`)。
テスト結果には「使うアンテナの本数ごと」の正解率も表示される。

**2 台の受信機の長さの違い**

2 台の USRP は受けたパケットの数が違うので、ドップラーデータの時間方向の長さが
揃わない (SHARP は 4 コアが同じパケットを受けるため常に一致していた)。
データセット作成時に、短いほうに合わせて末尾を切る (画面に表示される)。
速度軸の範囲 (取得レートで決まる) がファイル間で 10% 以上違うと警告を出すので、
そのときは手順4 の `--sub_sampling` でレートを揃える。

**出力**

| | |
|---|---|
| `doppler_traces/S1/train_antenna1,antenna2_E,L,W,R,J/` など | データセット (窓ごとのファイル) |
| `usrp_antenna1,antenna2_E,L,W,R,J_network.keras` | 学習したモデル (検証正解率が最良のエポック) |
| `outputs/complete_different_antenna1,antenna2_E,L,W,R,J_S2,S3.txt` | テスト結果。`CSI_network_metrics.py` で再表示できる |
| `plots/complete_different_..._max_merge.png` | 混同行列 (アンテナ統合後) |

**VS Code で実行する**

リポジトリのフォルダを VS Code で開き、Python 拡張機能を入れて、上の `pip install`
をした環境をインタープリタに選ぶ (コマンドパレット → `Python: Select Interpreter`)。
あとは次のどちらかで実行できる。

- ターミナル (`` Ctrl+@ ``) で上のコマンドを実行する
- 「実行とデバッグ」(`Ctrl+Shift+D`) で `SHARP 0. 全部まとめて実行` を選んで
  `Ctrl+F5` (デバッグなしで実行)。段階ごとに回すなら `SHARP 1.` 〜 `SHARP 5.` を順に。
  起動時に使うアンテナ (と活動ラベル) を聞かれる。設定はリポジトリ直下の
  `.vscode/launch.json` にあり、データの場所やセッション名を変えるときはその
  `args` を書き換える
- VS Code に Claude Code の拡張機能を入れると、Claude Code が PC 上で直接
  コマンドを実行できる。「SHARP-USRP の学習を S1 で回して S2,S3 でテストして」と
  頼めば、リポジトリ直下の `CLAUDE.md` を読んで run_har.py を回し、結果をまとめる

**表示されるが無視してよい警告**

- `The calling iterator did not fully read the dataset being cached` —
  TensorFlow のキャッシュの既知の警告。キャッシュは正しく作られる
- `shuffle=True was passed, but will be ignored` — データセット側でシャッフル
  しているので問題ない

### 受信レートの低下を模擬する (packet_loss.py)

実環境では他の通信・干渉・混雑などで CSI を受信できないパケットが出る。
位相サニタイゼーション済みの CSI (`processed_phase/`) のパケットを「受信 / 欠落」
に振り分けるマスクを掛けて、受信レートが下がった状況を疑似的に作る。
受信が止まる時間の長さと規則性で 4 つのパターンを用意しており、どれも
**欠落率 `--loss_rate` (0〜1) で強さを変えられる**。

| `--loss` | 受信の様子 | 想定する原因 | 強さを決めるオプション |
|---|---|---|---|
| `random` | 1 パケットずつ散発的に欠ける | 低 SNR (距離・壁)、他の端末との衝突 | `--loss_rate` |
| `burst` | 一定時間まとめて途切れ、その後回復する | 他の通信によるチャネル占有、干渉、人体による遮蔽 | `--loss_rate`, `--burst_ms` (途切れの平均長) |
| `periodic` | 一定の周期でまとめて途切れる | 端末・AP の定期スキャン | `--period_s` (周期) と `--loss_rate` (または `--burst_ms`) |
| `sustained` | 途切れはしないが受信レートが下がり続ける | 混雑で送信機が CSMA/CA で待機し続ける | `--loss_rate` (レートが 1 - 欠落率 倍になる) |

- `burst` は途切れる時間・受信する時間の長さが指数分布 (Gilbert-Elliott モデルを
  時間で表したもの)。平均の途切れが `--burst_ms`、長い目で見た欠落の割合が
  `--loss_rate` になる。`--burst_dist fixed` で途切れの長さを固定できる
- `periodic` の 1 回の途切れは `欠落率 x 周期` (例: `--period_s 30 --loss_rate 0.01`
  なら 30 秒ごとに 300 ms)。`--burst_ms 300` と直接書いてもよい
- `--loss_start_s 20 --loss_duration_s 30` のように、測定の一部の区間だけを
  低下させることもできる (先頭のパケットからの秒数)
- 欠け方は乱数で決まるが、ファイル名と `--loss_seed` から作るので再実行で同じになる
- 2 台の受信機 (`antenna1`, `antenna2`): `random` は受信側の要因なので受信機ごとに
  独立に欠け、それ以外は送信側の要因 (送られなかったパケット) なので 2 台で同じ
  パケットが欠ける。`--loss_antennas shared / independent` で固定できる

欠落したパケットは既定では取り除く (`--loss_fill drop`)。実際に受信が欠けたときに
パイプラインが受け取るデータと同じで、`time_sec` も欠落後のものになる。
`hold` (直前の受信で埋める) と `interp` (前後から線形補間) も選べる。

**欠落率ごとの正解率を比べる (run_har.py --loss)**

```powershell
cd SHARP-USRP\Python_code

# S1 (欠落なし) で学習し、S2,S3 に burst の欠落を入れて欠落率ごとにテストする
python run_har.py --train S1 --test S2,S3 --noise_level -3.0 --despike `
    --loss burst --loss_rate 0.1,0.2,0.4 --burst_ms 100

# 学習側にも同じ欠落を入れる (欠落率ごとに学習し直す)
python run_har.py --train S1 --test S2,S3 --noise_level -3.0 --despike `
    --loss sustained --loss_rate 0.3,0.6 --loss_on both
```

1. `processed_phase/` から、欠落なしで全セッションのドップラーデータを計算する
   (`doppler_traces_loss/none/`)。これが基準
2. 欠落率ごとに、テストのセッションへ欠落を入れて計算する
   (`doppler_traces_loss/burst_r0.2_b100ms/` など)
3. 欠落なしのデータで 1 回学習し、条件ごとにテストする

`outputs/har_summary_<日時>.md` の先頭に、条件ごとの 実際の欠落率・受信レート・
最長の途切れ・正解率 を並べた表が出る。混同行列の図は条件ごとに
`plots/complete_different_..._S2,S3_burst_r0.2_b100ms_max_merge.png` のように分かれる。

- 位相サニタイゼーション済みの CSI (`processed_phase/S1_*.mat` など) が必要。
  `--csi` と一緒に使えば、先にそれを作ってから欠落の模擬に進む
- ドップラー計算の設定 (`--noise_level`, `--despike`, `--remove_static` など) は
  基準と欠落ありの両方に同じものを使う。以前の `doppler_traces/` と設定が違えば
  警告を出す

**単体で使う**

```powershell
# processed_phase の .mat に欠落を入れて別のフォルダに書き出す (受信の様子の図も出す)
python packet_loss.py ./processed_phase/S2_*.mat --out_dir ./processed_phase_loss/ `
    --loss burst --loss_rate 0.2 --burst_ms 100 --preview

# ドップラー計算で直接入れる (--only で対象のファイルを絞れる)
python CSI_doppler_computation.py ./processed_phase/ . ./doppler_traces_loss/test/ 10 10 31 1 -3.0 `
    --despike --only S2_,S3_ --loss periodic --period_s 30 --burst_ms 300
```

書き出した `.mat` には元のパケットごとの受信 (1) / 欠落 (0) が `loss_mask` として、
欠落の実績 (欠落率、連続した欠落の長さ、受信レート) が `<名前>_loss.json` に入る。
ドップラー計算で入れた場合は `_meta.json` の `packet_loss` に残る。

**注意**

- `drop` では受信したパケットだけを並べて STFT に掛けるので (既定の `--resample none`)、
  途切れた時間は詰められる。また `sustained` で受信レートが下がると Tc が延び、
  ドップラーの速度軸の範囲 (`v_max`) が狭くなる。これらは実際にレートが落ちたときに
  このパイプラインで起きることそのもの。途切れの時間を正しく扱いたい場合は
  `--resample nudft` を試す
- 欠落はパケット数を減らすので、ドップラーの時間窓の数も減る。測定が短いと
  学習・テスト用の窓が作れなくなることがある (`--window` を小さくする)
- 位相サニタイゼーションの後で欠落させている。サニタイゼーションはパケットごとの
  処理なので、欠落してから処理した場合とほぼ同じになる (前処理のサブキャリアごとの
  振幅の正規化だけは全パケットの平均を使うが、パケットが減っても平均はほぼ変わらない)

### MATLAB から変換する場合

Python を使わず MATLAB 内で手順0 を済ませたい場合:

```matlab
addpath('SHARP-USRP/matlab');
usrpCSItoSHARP('CSI/202608191054_OpenWrt-A_CSI.mat', ...
               'SHARP-USRP/Python_code/input_files/exp01.mat');
```

その後、手順1 以降を Python で実行する。MATLAB 版と Python 版の出力が
ビット単位で一致することはテストで確認している (後述)。

> **注意**: MATLAB 側は必ず `-v7` 形式で保存している。Python の
> `scipy.io.loadmat` は v7.3 (HDF5) を読めないため、ここを `-v7.3` にすると
> 手順1 で失敗する。

## 大きいデータの扱い

デコーダの出力は 30 秒のキャプチャで 50MB を超え、長いものは 300MB 近くになる。
GitHub は 1 ファイル 100MB で拒否、50MB で警告を出すため、生データをそのまま
リポジトリに置くことはできない。

SHARP / SHARPax 本家も同じ方針で、`input_files/` は空のまま、データセットは外部
(パドヴァ大のリポジトリ / IEEE DataPort) に置いている。**コードは Git、データは外部**
というのがこの分野の標準である。

### 方針

| 用途 | 置き場所 |
|---|---|
| 生データの保管 | OneDrive / Google Drive / 大学のストレージ |
| 日常の解析 | データがある PC でローカル実行 (下記) |
| 共有・検証用の標本 | `slimCSI.m` で縮小してリポジトリへ |
| 論文公開時 | Zenodo / IEEE DataPort |

### 自分の PC で動かす

パイプラインはデータのある場所で動かせばよい。パスは任意の場所を指定できる。

```powershell
cd <repo>\SHARP-USRP\Python_code
python run_phase_sanitization.py "D:\IQ_csi\09272233_WAX202_CSI.mat" --name run2233
python CSI_doppler_computation.py ./processed_phase/ . ./doppler_traces/ 10 10 31 1 -1.5
python CSI_doppler_plot.py ./doppler_traces/ ./plots/
```

初回のみ: `pip install numpy scipy osqp h5py matplotlib`

### 標本を作る (slimCSI.m)

デコーダ出力の容量の内訳は、実測 (55.8MB, HE 4956 パケット) で次のとおり:

| 変数 | サイズ | 標本に必要か |
|---|---|---|
| `csiHE` | 18.4 MB | 必要 |
| `csi` | 18.4 MB | 不要 (`csiHE` の重複。`ResultCSI.m` 互換のための二重保存) |
| `csiNonHT` | 8.4 MB | 不要 (SHARP の処理では使わない) |

`slimCSI.m` は使う形式ひとつ分だけを残し、CSI を `single` 精度に落とす。

```matlab
addpath('SHARP-USRP/matlab');
slimCSI('D:\IQ_csi\09272233_WAX202_CSI.mat', 'CSI_ax\09272233_sample.mat');

% 長時間キャプチャから先頭 8000 パケットだけ抜く場合
slimCSI('D:\IQ_csi\09272233_WAX202_CSI.mat', 'CSI_ax\09272233_sample.mat', ...
        'MaxPackets', 8000);
```

実測で **55.8MB → 8.8MB (84% 削減)**。縮小後のファイルから手順0 を通した結果は、
元ファイルからの結果と一致する (差は `single` の丸め誤差のみ、相対 8.9e-8)。

出力は `-v7` 形式なので、Python 側は `h5py` 無しでも読める。

## 規格ごとの構成

```bash
python3 Python_code/wifi_config.py   # 一覧を表示
```

| | VHT20 (Wi-Fi 5) | HE20 (Wi-Fi 6) | 参考: SHARP オリジナル (80MHz VHT) |
|---|---|---|---|
| FFT 長 | 64 | 256 | 256 |
| サブキャリア間隔 Δf | 312.5 kHz | 78.125 kHz | 312.5 kHz |
| 使用サブキャリア | 56 本 (k=±1..28) | 242 本 (k=±2..122) | 242 本 |
| ヌル (ガード+DC) | 8 本 | 14 本 | 14 本 |
| 再構成の出力幅 | 57 | 245 | 245 |
| 遅延分解能 (=1/帯域) | 50 ns | 50 ns | 12.5 ns |
| 最大非曖昧遅延 (=1/Δf) | 3.2 µs | 12.8 µs | 3.2 µs |

**HE20 は FFT 長・ヌル配置がオリジナル SHARP の 80MHz VHT と完全に同一**
(どちらも 256 点、ヌルは `[0..5, 127,128,129, 251..255]`)。つまり Wi-Fi 6 20MHz は
オリジナルのパラメータがほぼそのまま通用する、都合の良いケースである。

## SHARP オリジナルからの変更点

変更点の詳細（オリジナルのコードとの対比、変えていない部分の明示、
数値の根拠）は **[CHANGES_FROM_SHARP.md](CHANGES_FROM_SHARP.md)** にまとめてある。
以下はその要約。

| 変更 | 理由 |
|---|---|
| FFT 長・Δf・ヌル配置を `wifi_config.py` に分離 | オリジナルは 80MHz 決め打ち (`F_frequency = 256`, `delete_idxs` 直書き) |
| 再構成の切り出しを `[6:-5]` 固定から一般化 | 帯域によって使用サブキャリアの範囲が変わるため |
| ストリーム数を固定 4 から `n_ss * n_core` へ | オリジナルは `for stream in range(0, 4)` と Nexmon の 4 コア前提。SISO では 1 |
| Nexmon 固有の符号反転を既定で無効化 | `signal_stream[:, 64:] = -signal_stream[:, 64:]` は Nexmon の CSI 規約に対する補正。MATLAB WLAN Toolbox のチャネル推定値には不要 (`--nexmon_sign_flip` で有効化可) |
| `subcarriers_space` / `delta_t_refined` を規格ごとの既定値 + CLI 指定に | 下記「20MHz での注意」を参照 |
| 出力先サブディレクトリを `--label` で指定 | オリジナルはファイル名先頭3文字を活動ラベルとして使う実装。USRP のファイル名は日時始まりのため不適合 |
| `np.linalg.lstsq` に `rcond=None` を明示 | 新しい NumPy での警告回避 |

### 20MHz での注意

位相サニタイゼーションの最適化は「観測 = 使用サブキャリア数」に対して
「未知数 = 遅延グリッドの列数」を解く問題である。80MHz (242 本) 前提の既定値を
20MHz にそのまま持ち込むと、VHT20 では観測が 56 本しかないため劣決定が過度になる。
そこで規格ごとに既定値を変えている:

| | VHT20 | HE20 |
|---|---|---|
| `subcarriers_space` (間引き) | **1** (全 56 本を使用) | 2 (オリジナル同様) |
| `delta_t_refined` (精細グリッド刻み) | **10 ns** | 5 ns (オリジナル同様) |
| 結果の観測数 vs 精細グリッド列数 | 56 vs 45 | 121 vs 90 |

VHT20 で刻みを 10 ns にしているのは、20MHz の遅延分解能が 50 ns である以上、
5 ns 刻みは分解能を超えて細かすぎるという理由もある。いずれも
`--subcarriers_space` / `--delta_t_refined` で上書きできる。

**帯域による本質的な制約**: 20MHz の遅延分解能は 50 ns (= 距離にして約 15 m)
であり、80MHz の 12.5 ns に比べて 4 倍粗い。位相オフセットの除去自体は問題なく
機能するが、近接した多重波の分離能力は原理的に劣る。これは実験条件が 20MHz で
ある以上避けられない。

## 検証

### 1. 変換の可逆性

`fftshift` で中心寄せに戻したとき、使用サブキャリアの値が元の CSI と
**完全一致** (最大絶対差 0)、ヌル位置は厳密に 0。

### 2. 実データでの位相サニタイゼーション効果

`CSI/202608191054_OpenWrt-A_CSI.mat` (Wi-Fi 5, 20MHz, Ch36, 850 パケット, 5 秒):

| | パケット間の位相差のばらつき |
|---|---|
| 生 CSI | 1.96 rad (ほぼランダム) |
| サニタイズ後 | **0.16 rad** (約 1/12) |

平均振幅プロファイルの相関は 0.9992 で、振幅情報は保たれている。
残る 0.16 rad は実際のチャネル変動と雑音によるもの。

処理時間は 850 パケットで約 13 秒。

### 3. Wi-Fi 6 実データでの確認

`CSI_ax/202609141849_WAX202_CSI.mat` (Wi-Fi 6, 20MHz, Ch36, 4956 パケット, 30 秒):

| | 結果 |
|---|---|
| サブキャリア構成 | `decodeIQ_HE.m` が実測した 242 本 (k=-122..122) が HE20 の想定と**完全一致** |
| パケット間の位相ばらつき | 1.82 rad → **0.020 rad (90.5 分の1)** |
| 平均振幅プロファイルの相関 | 0.9990 |

VHT20 (12.2 分の1) より改善幅が大きいのは、サブキャリアが 242 本と VHT20 の 4.3 倍あり、
多重波推定の条件が良いため。処理時間は 4956 パケットで約 5 分。

### 4. 合成データでの厳密な確認 (VHT20 / HE20 両方)

既知の静的多重波にパケットごとのランダム位相・時刻オフセットを載せた合成データでは、
真のチャネルが静的なので理想的にはばらつきが 0 になるはず:

| | 位相ばらつきの改善 | 振幅相関 |
|---|---|---|
| VHT20 | 105 倍 (1.758 → 0.017 rad) | 0.9999 |
| HE20 | 125 倍 (1.765 → 0.014 rad) | 1.0000 |

```bash
python3 Python_code/tests/test_pipeline.py
```

### 5. MATLAB 版と Python 版の一致

同じ入力に対して両者の `csi_buff` がビット単位で一致することを確認
(実データ・合成データとも最大絶対差 0)。GNU Octave で実行するため MATLAB 本体は不要。

```bash
python3 Python_code/tests/test_matlab_parity.py
```

### 6. 実データのドップラーマップ

`202608191054` (Wi-Fi 5, 20MHz, Ch36, 850 パケット, 5 秒) から得たマップは、
0 m/s に静止成分が支配的で、0.9 / 1.65 / 2.45 秒付近に ±2 m/s 程度の
ドップラー拡がりが現れる、という物理的に妥当な構造を示した。
`none` と `nudft` の結果は相関 0.9996 で一致。

## 今後の課題

### 活動認識の実データでの検証

学習側 (手順6〜8) は移植済みで、合成ドップラーデータでは S1 で学習 → S2・S3 で
テストの流れが通り、正しく分類できることを確認している
(`python3 Python_code/tests/test_har.py`)。実測データでの精度はまだ確かめていない。

### パケット取得レートの安定化

実測 (`202608191054`) は平均 170 pkt/s、間隔 0.60〜12.06 ms、変動係数 0.15。
この程度なら既定の `none` で理想とほぼ同等であることを確認したが、より速い動きを
扱う場合や、間隔のばらつきがこれより大きい測定では `--resample nudft` を使うか、
AP 側から一定レートでトラフィックを流す (ping の間隔固定など) ほうがよい。

### 動きのある Wi-Fi 6 測定

Wi-Fi 6 の実データ (`CSI_ax/202609141849_WAX202_CSI.mat`) は位相サニタイゼーション
まで問題なく通ったが、**測定環境がほぼ静止していた**ため、ドップラーマップには
0 m/s の静止成分しか現れなかった (時間平均からの CFR のずれが 2.44%。
動きのある Wi-Fi 5 の測定では 62.57%)。活動認識に進むには、人が歩く・走るなどの
動作を含む Wi-Fi 6 測定が必要。

## ファイル構成

```
SHARP-USRP/
├── README.md
├── LICENSE                     GPL v3 (SHARP の派生のため)
├── matlab/
│   ├── usrpCSItoSHARP.m        手順0 の MATLAB 版
│   └── slimCSI.m               .mat を解析に必要な変数だけに縮小
└── Python_code/
    ├── wifi_config.py          規格ごとの OFDM 構成定義
    ├── usrp_to_sharp.py        手順0: 形式変換
    ├── CSI_phase_sanitization_signal_preprocessing.py    手順1
    ├── CSI_phase_sanitization_H_estimation.py            手順2
    ├── CSI_phase_sanitization_signal_reconstruction.py   手順3
    ├── CSI_doppler_computation.py  手順4: ドップラースペクトル
    ├── CSI_doppler_plot.py         手順5: ドップラーマップの描画
    ├── run_phase_sanitization.py   手順0〜3の一括実行 (1ファイル)
    ├── run_batch.py                手順0〜5の一括実行 (複数ファイル)
    ├── run_har.py                  手順6〜8 (--csi で 0〜8) の一括実行と結果のまとめ
    ├── packet_loss.py              受信レートの低下 (パケット欠落) の模擬
    ├── CSI_doppler_create_dataset_train.py  手順6: 学習用データセット
    ├── CSI_doppler_create_dataset_test.py   手順6: テスト用データセット
    ├── CSI_network.py              手順7: 学習
    ├── CSI_network_test.py         手順8: テスト
    ├── CSI_network_metrics.py      評価結果の表示
    ├── CSI_network_metrics_plot.py 混同行列の描画
    ├── dataset_utility.py          ファイル名の解釈・アンテナの束ね方・窓切り
    ├── metrics_utility.py          アンテナごとの予測の統合と指標
    ├── network_utility.py          SHARP 由来 (無変更)
    ├── optimization_utility.py     SHARP 由来 (無変更)
    ├── pipeline_meta.py            段間の設定引き継ぎ
    └── tests/
        ├── make_synthetic_usrp_mat.py  合成データ生成
        ├── test_pipeline.py            パイプライン全体のテスト
        ├── test_doppler.py             ドップラー計算の検証
        ├── test_har.py                 データセット作成〜学習〜テストの検証
        ├── test_packet_loss.py         パケット欠落の模擬の検証
        └── test_matlab_parity.py       MATLAB/Python 一致テスト
```

## ライセンス

本リポジトリは SHARP (Francesca Meneghello, GNU GPL v3) の派生物であり、
同じく **GNU GPL v3** で配布する。`optimization_utility.py` はオリジナルの
コードをそのまま利用している。

参考文献:

- F. Meneghello et al., "SHARP: Environment and Person Independent Activity
  Recognition with Commodity IEEE 802.11 Access Points," IEEE Transactions on
  Mobile Computing, 2022.
- F. Meneghello et al., "Toward Integrated Sensing and Communications in
  IEEE 802.11bf Wi-Fi Networks," IEEE Communications Magazine, 2023.
- オリジナル実装: https://github.com/francescamen/SHARP ,
  https://github.com/francescamen/SHARPax
