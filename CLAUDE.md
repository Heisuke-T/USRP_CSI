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
- 必要なパッケージ: `pip install numpy scipy osqp h5py matplotlib tensorflow scikit-learn`

## テスト

```powershell
cd SHARP-USRP/Python_code
python tests/test_har.py        # データセット作成〜学習〜テスト、run_har.py (合成データ)
python tests/test_pipeline.py   # 位相サニタイゼーション
python tests/test_doppler.py    # ドップラー計算
```

## 約束

- ブランチは `claude/usrp-wifi-iq-capture-ur0hq1` の 1 本だけを使う (ユーザーの希望)。
  新しいブランチを作らない
- 測定データ・中間生成物・モデル・結果 (`.mat`, `doppler_traces/`, `*.keras`, `outputs/`)
  はコミットしない (`.gitignore` 済み)
