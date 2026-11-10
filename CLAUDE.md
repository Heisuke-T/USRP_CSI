# CLAUDE.md

このリポジトリで作業する Claude Code 向けのメモ。ユーザーとのやりとりは日本語で行う。

## 構成

- 直下の `.m` (MATLAB): USRP B205mini-i での IQ 取得 (`captureIQ*.m`)、Wi-Fi 5/6 の復号
  (`decode_VHT_v2.m`, `decodeIQ_HE*.m`)、2 台の受信機の時間合わせ (`alignCSI.m`)
- `SHARP-USRP/`: 復号した CSI から 位相サニタイゼーション → ドップラー → 活動認識
  (SHARP の移植, Python)。詳細は `SHARP-USRP/README.md`、SHARP との違いは
  `SHARP-USRP/CHANGES_FROM_SHARP.md`
- `external/SHARP`, `external/SHARP-ax`: 参照用のオリジナル (サブモジュール)。編集しない

## 活動認識を最後まで回す

測定データ (デコード済み CSI の `.mat`) はリポジトリの外にある。セッションごとの
フォルダに `<活動>_<アンテナ>.mat` の名前で置かれている (例: `E:/Dataset_CSI/S1/W_antenna1.mat`)。
場所が分からなければユーザーに聞く。

```powershell
cd SHARP-USRP/Python_code

# ドップラーデータ (doppler_traces/) ができている場合 (数分〜)
python run_har.py --train S1 --test S2,S3

# デコード済み CSI から全部 (位相サニタイゼーションが約 5 分 / 5000 パケットなので数時間)
python run_har.py --train S1 --test S2,S3 --csi E:/Dataset_CSI --noise_level -3.0 --despike

# アンテナ 2 本と 1 本の両方で回して比べる
python run_har.py --train S1 --test S2,S3 --antennas antenna1,antenna2 --antennas antenna1
```

- 結果は `outputs/har_summary_<日時>.md` (正解率・F 値の表と混同行列)。ユーザーには
  この内容を示し、`plots/complete_different_*_max_merge.png` (混同行列の図) の場所も伝える
- 時間が掛かるのでバックグラウンドで実行し、`har_<日時>.log` で進み具合を確かめる
- 失敗したら止まって失敗した工程を表示する。ログを読んで原因を調べてから報告する
- ドップラー計算の設定 (`--noise_level`, `--despike`, `--remove_static` など) は
  S1〜S3 で揃える。以前の実行例は `--noise_level -3.0 --despike` (変えるならユーザーに確認する)
- 受信レートの低下を模擬して欠落率ごとの正解率を比べる (`packet_loss.py`, 要 `processed_phase/`):
  `python run_har.py --train S1 --test S2,S3 --noise_level -3.0 --despike --loss burst --loss_rate 0.1,0.2,0.4 --burst_ms 100`
  (`--loss` は random / burst / periodic / sustained。詳細は `SHARP-USRP/README.md`)
- 必要なパッケージ: `pip install numpy scipy osqp h5py matplotlib tensorflow scikit-learn xgboost`

## ドップラー → ENMO の回帰

`regress_enmo.py`: 線形回帰・ランダムフォレスト・XGBoost・MLP・SVR・ガウス過程回帰・CNN で、ドップラー (既定は特徴量 F:
動いている速度ビン (|v| ≥ 0.65 m/s) で足したドップラーの時間微分 (Savitzky-Golay、窓 0.2 秒) の 1 秒平均。F1 / F2a / F2b。
前に作った特徴量 A: 平均スペクトル、
B: 物理特徴量。C: 速さの時間微分は実データで効かなかったので既定では使わない。D: 1 秒の中の時間変化、
E: 強さをそろえたスペクトルは `--features` で指定。`B+D@1` で前後 1 区間の特徴量も並べる。
`--importance` で決定木の特徴量の重要度) から
腰の ENMO (加速度計算ソフトが出す max(合成加速度 − 1 g, 0)、1 秒ごと) を推定して比べる。`--target body` で
全身の ENMO (腰・大腿・下腿・手首の 4 台の ENMO の合計、列 `ENMO_body(mG)`。4 台そろった試行だけ) を正解にする。入力は
`regression_input/<試行名>/` (置き方は `regression_input/README.md`、データはコミットしない。
`regression_input/その他/` はユーザーの希望で読まない: `--exclude` の既定)、
結果は `outputs/regression_<日時>/`。評価は試行ごとに分ける (ランダムに分けない)。
`--split A1:A2` で学習とテストを指定、`--cnn_seeds 5` で CNN を 5 回学習、`--cnn_study` で CNN が効く理由の確認
(時間の順番の入れ替え・時間平均・1 次元 CNN・前後 1 秒・Grad-CAM)。前後 2 秒の CNN は `--models cnn5s`。
`--gain_jitter 0,3,6` で CNN の学習のときだけ動いている速度ビンの強さを区間ごと・アンテナごとに最大 ±N dB ゆらす
(測った回ごとの強さの違いに強くする。A1 だけ動いている成分が強く、A2〜A4 で学習すると A1 を高く出したため)

結果の受け渡し: ユーザーはワークステーションで回したあと `python share_results.py` を実行する。
結果の軽いファイル (log.txt・metrics*.csv・predictions.csv・comparison.png・cnn_gradcam_*.png など) が
`SHARP-USRP/Python_code/shared_results/<結果のフォルダ名>/` に入ってこのブランチにプッシュされるので、
「○○を見て」と言われたら `git pull` してそこを読む

## テスト

```powershell
cd SHARP-USRP/Python_code
python tests/test_har.py        # データセット作成〜学習〜テスト、run_har.py (合成データ)
python tests/test_pipeline.py   # 位相サニタイゼーション
python tests/test_doppler.py    # ドップラー計算
python tests/test_packet_loss.py  # 受信レートの低下 (パケット欠落) の模擬
python tests/test_doppler_variants.py  # 正規化を省いたドップラー (--save_unnormalized)
python tests/test_regress_enmo.py      # ドップラー → ENMO の回帰 (合成データ)
```

## 約束

- ブランチは `claude/usrp-wifi-iq-capture-ur0hq1` の 1 本だけを使う (ユーザーの希望)。
  新しいブランチを作らない
- 測定データ・中間生成物・モデル・結果 (`.mat`, `doppler_traces*/`, `*.keras`, `outputs/`)
  はコミットしない (`.gitignore` 済み)。例外は `share_results.py` が入れる `shared_results/` だけ
