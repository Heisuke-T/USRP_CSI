# 回帰モデルの入力フォルダ（regress_enmo.py）

置き方は 2 通りあります。どちらでも `regress_enmo.py` が自動で見つけます。試行名は、正解データのファイル名
（`<試行名>_power_timeseries.csv`）から決まります。

## 置き方 1: データセットごとにまとめる（一括処理の結果をそのまま使う）

```
regression_input/
└── A1/
    ├── imu/                                 ← 加速度計算ソフトの一括処理で「正解データ（親フォルダ）」にここを指定
    │   ├── A1_XR1/  A1_XR1_power_timeseries.csv（と _windows_1s/5s.csv）   ← <試行名>/imu/ の中でもよい
    │   ├── A1_XR2/  …
    │   …
    └── doppler_traces_nowinnorm/            ← SHARP-USRP の doppler_traces_nowinnorm/ から A1 の分をまとめてコピー
        ├── A1_XR1_antenna1_…_stream_0.txt（_time.npy・_meta.json も）
        ├── A1_XR1_antenna2_…_stream_0.txt
        …
```

- 正解データとドップラーは **名前** で対応づけます（ドップラーのファイル名の antenna より前の部分と、試行名が同じ・似ている）
- 名前が対応しないとき（CSI の名前と IMU の名前が違うとき）は、加速度計算ソフトの一括処理が保存した
  **`batch_pairs_<日時>.csv`**（IMU と CSI の組み合わせ）を入力フォルダのどこかに置くか、`--pairs` で指定すると、それで対応づけます
- 実行すると、最初に「試行: ○ 本」と組み合わせの一覧がログに出ます。違っていないか確かめてください

## 置き方 2: 試行ごとのフォルダ

試行ごとにフォルダを作り、次のようにファイルを置きます。
ここに置いたデータは Git に入りません（この README だけが入ります）。

```
regression_input/
├── T1/
│   ├── T1_A_R_power_timeseries.csv          ← 正解データ（加速度計算ソフトの出力。_windows_1s/5s.csv も一緒でよい）
│   ├── doppler_traces_nowinnorm/            ← SHARP-USRP の doppler_traces_nowinnorm/ から、この試行の分をコピー
│   │   ├── T1_antenna1_stream_0.txt
│   │   ├── T1_antenna1_stream_0_time.npy
│   │   ├── T1_antenna1_stream_0_meta.json
│   │   ├── T1_antenna2_stream_0.txt
│   │   ├── T1_antenna2_stream_0_time.npy
│   │   └── T1_antenna2_stream_0_meta.json
│   ├── doppler_traces/                      ← （任意）通常の版。3 つの版を比べるとき
│   └── doppler_traces_raw/                  ← （任意）割り算なしの版。3 つの版を比べるとき
├── T2/
│   └── …（同じ構成）
…
```

- ドップラーは、ファイル名に `antenna1` / `antenna2` が入った `.txt` を 1 つずつ。`_time.npy` と `_meta.json` も必ず一緒に
- 加速度計算ソフトの「正解データの保存先」にこの試行のフォルダ（例 `regression_input/T1`）を指定すると、
  計算のたびに自動でここに保存されます。一括処理なら親フォルダ `regression_input` を指定すると、`<IMU の名前>/` に入ります
  （前の置き方の `<試行名>/imu/` に置いても読めます）
- IMU は `*_power_timeseries.csv`（ENMO の列 `ENMO(mg)` が入った版。6 面キャリブレーションを指定して計算したもの）。
  なければ `*_windows_1s.csv` / `*_windows_5s.csv` でも可
- ドップラーと IMU は、どちらも alignCSI.m でそろえた同じ 80 秒の CSI から作ったものを使う（時間軸が同じになる）

実行:

```
cd SHARP-USRP/Python_code
python regress_enmo.py                  # nowinnorm 版で 5 モデル、1 秒と 5 秒の区間
python regress_enmo.py --doppler all    # 3 つの版それぞれで比べる
```

結果は `outputs/regression_<日時>/`（比較の表 `metrics.csv`・図 `comparison.png`・モデルごとの予測の図 `pred_*.png` など）。
