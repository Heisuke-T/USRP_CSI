# 回帰モデルの入力フォルダ（regress_omron.py）

試行ごとにフォルダを作り、次のようにファイルを置きます。フォルダ名（例 `T1`）が試行名になります。
ここに置いたデータは Git に入りません（この README だけが入ります）。

```
regression_input/
├── T1/
│   ├── imu/
│   │   └── T1_A_R_power_timeseries.csv      ← 仕事率計算ソフトの出力（output/power/<名前>/ にある）
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
- IMU は `*_power_timeseries.csv`（オムロン式の列 `omron_acc(mG)` が入った最新版）。なければ `*_power_windows.csv` でも可
- ドップラーと IMU は、どちらも alignCSI.m でそろえた同じ 80 秒の CSI から作ったものを使う（時間軸が同じになる）

実行:

```
cd SHARP-USRP/Python_code
python regress_omron.py                 # nowinnorm 版で 5 モデル
python regress_omron.py --doppler all   # 3 つの版それぞれで比べる
```

結果は `outputs/regression_<日時>/`（比較の表 `metrics.csv`・図 `comparison.png`・モデルごとの予測の図 `pred_*.png` など）。
