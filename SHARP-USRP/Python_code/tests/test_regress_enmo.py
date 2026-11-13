"""
    regress_enmo.py の検証 (合成データ)。

    5 試行ぶんの入力フォルダを作る。各試行の中で動きの強さ s(t) が変わり (試行ごとに違う運動)、
      ドップラー: 足踏みの周期で速さが揺れる反射。速度の広がりと強さが s(t) に比例
                  (アンテナ 2 本、時刻は alignCSI の時間軸、_meta.json に time_origin_s)
      IMU:       ENMO = 60 * s(t) + 雑音 [mG] (加速度計算ソフトと同じ列の _power_timeseries.csv)
    これで regress_enmo.py を 1 秒と 5 秒の区間で回し、
      1. 3 つのドップラーの版・5 モデル + 基準 (平均値) が最後まで動き、表と図ができること
      2. 物理特徴量 B の線形回帰などが、平均値で予測する基準よりはっきり良いこと (R² > 0.5)
      3. 区間が加速度計算ソフトと同じ 0, 1, 2 … 秒の区切りになり、古い形式の meta (time_origin_s なし) でも動くこと
      4. 特徴量 C (速さの時間微分): 足踏みのテンポが試行ごとに違い、ENMO = 速さ x テンポ (加速度) のとき、
         線形回帰で B に C を足すと はっきり良くなること (速さだけでは当たらない分を C が拾う)
      6. CNN の確認 (--cnn_study, --cnn_seeds): 理由を確かめる CNN 4 種・乱数の種ごとの結果・Grad-CAM の図ができること
      7. SVR とガウス過程回帰 (特徴量 B) も基準よりはっきり良いこと
      8. 特徴量 D・E: どちらも動き、試行ごとに受信の強さが違うとき E (強さをそろえたスペクトル) が A より良いこと
         (ランダムフォレストで比べる。線形回帰は強さの違いを係数で打ち消せるので差が出にくい)
      9. 特徴量の重要度 (--importance) と前後の区間の特徴量 (B+D@1) が出ること
     10. CNN の強さのゆらぎ (--gain_jitter): 学習のときだけ、動いている速度ビンに区間ごと・アンテナごとの量を足し
         (静止ビンと区間の印のチャネルはそのまま)、テストでは何もしないこと。0 dB と 3 dB の両方の結果が出ること
     11. MAPE: 正解が --mape_min 未満の区間を除いて計算し、表 (metrics.csv) に MAPE(%) の列が出ること
     12. 加速度計を入力にする特徴量（腰 W・WS、手首 R・RS、--target body）: 全身 ＝ 腰と手首の和の合成データで、腰だけ・
         手首だけより W+R がよく当たり、ドップラーの特徴量と同じ表に並ぶこと
     13. ドップラーの図そのもの（特徴量 M）と PCA・PLS 回帰: --pca auto で A・M のときだけ PCA がかかり（成分数がログに出る）、
         PLS 回帰の成分数が学習データの中で選ばれ、係数の表と図ができ、どれも基準よりはっきり良いこと
     14. 加速度計の ENMO の波形の CNN（--models cnn_imu,cnn_imu3s、--imu_part waist / wrist）が最後まで動き、
         入力が 1 秒 50 点・前後 1 秒で 150 点（＋区間の印）になり、ドップラーの CNN と同じ表に並ぶこと
      5. regression_input/A1 と A2 を並べて置いたとき（「その他」フォルダの中は読まない）: データセットをまたいで名前で組にしないこと、
         --split で「学習 A1 → テスト A2」「学習 A1+A2_XR1 → テスト A2_XR2」が計算できること
    を確かめる。

    実行:  python3 tests/test_regress_enmo.py   (CNN は TensorFlow があるときだけ)

    Copyright (C) 2026 Heisuke Takeda
    Released under the GNU GPL v3 (see LICENSE).
"""

import csv
import json
import os
import pickle
import shutil
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import regress_enmo as ro  # noqa: E402

N_FFT, DV = 100, 0.05      # velocity bins, bin width [m/s]
FRAME_DT = 1 / 150.0       # Doppler frame interval [s]


def intensity(t, k):
    """Motion intensity of trial k: a different pattern in each trial. / 試行ごとに違う強さの変化"""
    rng = np.random.default_rng(100 + k)
    knots = np.linspace(0, 85, 9)
    vals = rng.uniform(0.3, 1.5, len(knots))
    return np.interp(t, knots, vals)


def make_trial(root, k, old_meta=False):
    name = f"T{k + 1}"
    rng = np.random.default_rng(k)
    d = os.path.join(root, name)
    # ground truth directly in the trial folder; trial 4 uses the old layout <trial>/imu/
    # 正解データは試行のフォルダの直下。4 番目の試行だけ前の置き方 <試行>/imu/
    truth_dir = os.path.join(d, "imu") if k == 3 else d
    os.makedirs(truth_dir, exist_ok=True)
    origin = 0.06
    t_frames = np.arange(0, 80.5, FRAME_DT)            # local frame times (first packet = 0)
    t_abs = t_frames + origin                          # on the CSI axis / CSI の時間軸
    s = intensity(t_abs, k)
    v = (np.arange(N_FFT) - N_FFT // 2) * DV
    swing = np.abs(np.sin(2 * np.pi * 0.9 * t_abs))    # stepping rhythm / 足踏みのリズム
    for variant, folder in ro.VARIANT_DIRS.items():
        os.makedirs(os.path.join(d, folder))
        for ant, gain in (("antenna1", 1.0), ("antenna2", 0.6)):
            width = 0.15 + 0.6 * s * swing
            S = (np.exp(-0.5 * (v[None, :] / width[:, None]) ** 2) * (s[:, None] * gain)
                 + 3.0 * gain * (np.abs(v) < 1e-9)[None, :])           # static path at 0 m/s
            S = S * (1 + 0.1 * rng.standard_normal(S.shape)) ** 2 + 1e-3
            if variant == "normalized":
                S = np.clip(S / S.max(axis=1, keepdims=True), 1e-4, None)
            elif variant == "raw":
                S = S * 50.0
            base = os.path.join(d, folder, f"{name}_{ant}_stream_0")
            with open(base + ".txt", "wb") as fp:
                pickle.dump(S, fp)
            np.save(base + "_time.npy", t_frames)
            meta = dict(delta_v_bin=DV, Tc=0.006, start=10, n_fft=N_FFT)
            if not old_meta:
                meta["time_origin_s"] = origin
            with open(base + "_meta.json", "w") as fp:
                json.dump(meta, fp)
    t50 = np.arange(0, 80.0, 0.02)
    en = 60 * intensity(t50, k) + rng.normal(0, 2.0, len(t50))
    with open(os.path.join(truth_dir, f"{name}_power_timeseries.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_csi(s)", "t_imu(s)", "ENMO(mg)"])   # the old unit spelling still reads / 前の単位の書き方も読める
        for a, b in zip(t50, en):
            w.writerow([f"{a:.4f}", f"{a + 40:.4f}", f"{b:.3f}"])


def check_feature_c(tmp):
    """ENMO proportional to acceleration (speed x cadence), cadence different in each trial.
    ENMO が加速度 (速さ x テンポ) に比例し、テンポが試行ごとに違うとき、特徴量 C が効くか"""
    import argparse
    root = os.path.join(tmp, "in_c")
    v = (np.arange(N_FFT) - N_FFT // 2) * DV
    for k, cad in enumerate((0.7, 1.0, 1.3, 1.6, 1.9, 2.2)):
        name = f"C{k + 1}"
        rng = np.random.default_rng(50 + k)
        d = os.path.join(root, name)
        folder = os.path.join(d, ro.VARIANT_DIRS["nowinnorm"])
        os.makedirs(folder)
        t_frames = np.arange(0, 80.5, FRAME_DT)
        amp = intensity(t_frames, k)                           # how big the steps are / 動きの大きさ
        swing = np.abs(np.sin(2 * np.pi * cad * t_frames))
        for ant, gain in (("antenna1", 1.0), ("antenna2", 0.6)):
            width = 0.15 + 0.6 * amp * swing
            S = (np.exp(-0.5 * (v[None, :] / width[:, None]) ** 2) * gain + 3.0 * gain * (np.abs(v) < 1e-9)[None, :])
            S = S * (1 + 0.1 * rng.standard_normal(S.shape)) ** 2 + 1e-3
            base = os.path.join(folder, f"{name}_{ant}_stream_0")
            with open(base + ".txt", "wb") as fp:
                pickle.dump(S, fp)
            np.save(base + "_time.npy", t_frames)
            with open(base + "_meta.json", "w") as fp:
                json.dump(dict(delta_v_bin=DV, n_fft=N_FFT, time_origin_s=0.0), fp)
        t50 = np.arange(0, 80.0, 0.02)
        en = 60 * intensity(t50, k) * cad + rng.normal(0, 2.0, len(t50))   # acceleration-like / 加速度に比例
        with open(os.path.join(d, f"{name}_power_timeseries.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_csi(s)", "t_imu(s)", "ENMO(mG)"])
            for a, b in zip(t50, en):
                w.writerow([f"{a:.4f}", f"{a + 40:.4f}", f"{b:.3f}"])
    res, out_dir = ro.run(argparse.Namespace(input=root, out=os.path.join(tmp, "out_c"), doppler="nowinnorm",
                                             models="linear", features="B,C,B+C", windows=[1.0], blocks=5,
                                             epochs=5, seed=0, no_plots=True, pairs=None))
    r2 = {x["features"]: x["R2"] for x in res if x["model"] == "linear"}
    with open(os.path.join(out_dir, "features_nowinnorm_1s.csv"), encoding="utf-8") as f:
        head = next(csv.reader(f))
    good = r2["B+C"] > r2["B"] + 0.1 and r2["C"] > 0.5 and "C_ant1_acc_v95_rms" in head
    print(f"  {'OK ' if good else 'NG '} 特徴量 C: ENMO が速さ x テンポのとき、線形回帰の R² "
          f"B {r2['B']:.3f} → B+C {r2['B+C']:.3f}（C だけ {r2['C']:.3f}）、特徴量の表に C の列")
    return good


def check_cnn_study(tmp):
    """--cnn_study and --cnn_seeds (short training: only that everything runs). / CNN の確認（動くことだけ）"""
    import importlib.util
    if not importlib.util.find_spec("tensorflow"):
        print("  (TensorFlow が無いので CNN の確認は飛ばします)")
        return True
    inp = os.path.join(tmp, "in_study")
    for k in range(4):
        make_trial(inp, k)
    out_dir = os.path.join(tmp, "out_study")
    import argparse
    res, _ = ro.run(argparse.Namespace(input=inp, out=out_dir, doppler="nowinnorm", models="cnn", features="B",
                                       windows=[1.0], blocks=5, epochs=3, seed=0, no_plots=False, pairs=None,
                                       split=["T1,T2,T3:T4"], cnn_study=True, cnn_seeds=2))
    got = {r["model"]: r for r in res}
    files = os.listdir(out_dir)
    study = ["cnn"] + ro.STUDY_MODELS
    good = (all(m in got for m in study) and all(got[m]["n_seeds"] == 2 and len(got[m]["seed_metrics"]) == 2
                                                 for m in study)
            and "cnn_seeds.csv" in files and any(f.startswith("cnn_gradcam_examples_") for f in files))
    # the 5 s input (2 s before and after) / 前後 2 秒の入力
    res5, _ = ro.run(argparse.Namespace(input=inp, out=out_dir + "_5s", doppler="nowinnorm", models="cnn5s",
                                        features="B", windows=[1.0], blocks=5, epochs=2, seed=0, no_plots=True,
                                        pairs=None, split=["T1,T2,T3:T4"], cnn_study=False, cnn_seeds=1))
    good &= any(r["model"] == "cnn5s" for r in res5)
    print(f"  {'OK ' if good else 'NG '} CNN の確認: {', '.join(sorted(got))}、乱数の種 2 回、Grad-CAM の図 "
          f"{[f for f in files if f.startswith('cnn_gradcam_')]}")
    return good


def check_gain_jitter(tmp):
    """--gain_jitter: the layer only moves the moving bins while training, and a run gives one result per value.
    強さのゆらぎ: 学習のときだけ動いている速度ビンが動き、値ごとに結果が出ること"""
    import importlib.util
    if not importlib.util.find_spec("tensorflow"):
        print("  (TensorFlow が無いので強さのゆらぎの確認は飛ばします)")
        return True
    import tensorflow as tf
    v = np.linspace(-2, 2, 21)
    m = ro.jitter_mask(v, 0.65, 3)                 # 2 antennas + the mark channel / アンテナ 2 本 + 区間の印
    layer = ro._gain_jitter_layer(tf, m * 0.3)
    x = np.zeros((64, 8, len(v), 3), np.float32)
    on = layer(x, training=True).numpy()
    off = layer(x, training=False).numpy()
    mov = np.abs(v) >= 0.65
    i = int(np.argmax(mov))                        # first moving bin / 最初の動いているビン
    good = (np.all(off == 0) and np.all(on[:, :, ~mov] == 0) and np.all(on[..., 2] == 0)
            and np.all(np.abs(on[:, :, mov, :2]) <= 0.3 + 1e-6) and on[:, :, mov, :2].std() > 0.05
            and np.allclose(on[:, 0, i, :2], on[:, -1, i, :2])                     # same over time / 時間で一定
            and abs(np.corrcoef(on[:, 0, i, 0], on[:, 0, i, 1])[0, 1]) < 0.5)       # antennas apart / アンテナ別
    import argparse
    inp = os.path.join(tmp, "in_jitter")
    for k in range(4):
        make_trial(inp, k)
    res, out_dir = ro.run(argparse.Namespace(input=inp, out=os.path.join(tmp, "out_jitter"), doppler="nowinnorm",
                                             models="cnn,cnn3s", features="B", windows=[1.0], blocks=5, epochs=3, seed=0,
                                             no_plots=False, pairs=None, split=["T1,T2,T3:T4"], gain_jitter="0,3"))
    got = sorted((r["model"], r["jitter"]) for r in res if r["model"] != "baseline")
    preds = [f for f in os.listdir(out_dir) if f.startswith("pred_")]
    good &= (got == [("cnn", 0.0), ("cnn", 3.0), ("cnn3s", 0.0), ("cnn3s", 3.0)]
             and sum("jitter3dB" in f for f in preds) == 2 and len(preds) == 4)
    try:
        ro.parse_jitter("-1")
        good = False
    except ro.RegError:
        pass
    print(f"  {'OK ' if good else 'NG '} CNN の強さのゆらぎ: 結果 {got}、図 {len(preds)} 枚")
    return good


def check_mape(tmp):
    """MAPE leaves out the near-zero truths and is written to metrics.csv. / MAPE: 小さい正解を除き、表に出ること"""
    m = ro.metrics([0.0, 10.0, 100.0, 200.0], [30.0, 40.0, 110.0, 180.0], mape_min=50)
    good = abs(m["MAPE"] - 10.0) < 1e-9 and m["MAPE_n"] == 2
    m0 = ro.metrics([100.0, 200.0], [100.0, 200.0], mape_min=50)
    good &= m0["MAPE"] == 0.0
    import argparse
    inp = os.path.join(tmp, "in_mape")
    for k in range(4):
        make_trial(inp, k)
    out_dir = os.path.join(tmp, "out_mape")
    res, _ = ro.run(argparse.Namespace(input=inp, out=out_dir, doppler="nowinnorm", models="linear", features="B",
                                       windows=[1.0], blocks=5, epochs=5, seed=0, no_plots=False, pairs=None,
                                       split=["T1,T2,T3:T4"], mape_min=20.0))
    with open(os.path.join(out_dir, "metrics.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    lin = [r for r in rows if r["model"].startswith("線形")][0]
    good &= ("MAPE(%)" in rows[0] and float(lin["MAPE(%)"]) > 0 and lin["MAPE_min(mG)"] == "20"
             and int(lin["MAPE_windows"]) <= sum(1 for _ in res[0]["y"]))
    print(f"  {'OK ' if good else 'NG '} MAPE: 例 {m['MAPE']:.1f} %（{m['MAPE_n']} 区間）、線形回帰 {float(lin['MAPE(%)']):.1f} %"
          f"（{lin['MAPE_windows']} 区間、正解 {lin['MAPE_min(mG)']} mG 以上）")
    return good


def check_waist_input(tmp):
    """Features W / WS (waist ENMO of the same window) predict the whole-body ENMO. / 腰の ENMO → 全身の ENMO"""
    import argparse
    inp = os.path.join(tmp, "in_waist")
    for k in range(4):
        make_trial(inp, k)
    for k in range(4):
        name = f"T{k + 1}"
        path = os.path.join(inp, name, "imu" if k == 3 else "", f"{name}_power_timeseries.csv")
        with open(path, encoding="utf-8") as f:
            rows = list(csv.reader(f))
        rng = np.random.default_rng(500 + k)
        col = [i for i, c in enumerate(rows[0]) if c in ("ENMO(mG)", "ENMO(mg)")][0]   # test files use the old name
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(rows[0] + ["ENMO_body(mG)", "ENMO_wrist(mG)"])
            for r in rows[1:]:   # wrist moves on its own; whole body = 3 x waist + 2 x wrist / 手首は別に動く。全身 = 腰と手首の和
                tt = float(r[0])
                wr = 80 * (1 + np.sin(2 * np.pi * tt / (17 + 3 * k))) + rng.normal(0, 5.0)
                w.writerow(r + [f"{3 * float(r[col]) + 2 * wr + rng.normal(0, 10.0):.3f}", f"{wr:.3f}"])
    out_dir = os.path.join(tmp, "out_waist")
    res, _ = ro.run(argparse.Namespace(input=inp, out=out_dir, doppler="nowinnorm", models="linear,gpr",
                                       features="B,W,WS,R,RS,W+R", windows=[1.0], blocks=5, epochs=5, seed=0, no_plots=False,
                                       pairs=None, split=["T1,T2,T3:T4"], target="body"))
    got = {(r["model"], r["features"]): r for r in res}
    with open(os.path.join(out_dir, "log.txt"), encoding="utf-8") as f:
        logtxt = f.read()
    good = (all(k in got for k in [("linear", "W"), ("gpr", "WS"), ("linear", "R"), ("gpr", "RS"), ("linear", "W+R")])
            and got[("linear", "W+R")]["R2"] > 0.95
            and got[("linear", "W+R")]["R2"] > max(got[("linear", "W")]["R2"], got[("linear", "R")]["R2"]) + 0.05
            and "特徴量 W・WS" in logtxt and "特徴量 R・RS" in logtxt)
    try:   # WS needs the 50 Hz file; W alone with the waist target is allowed but noted / 腰が正解なら注意だけ
        ro.run(argparse.Namespace(input=inp, out=out_dir + "_w", doppler="nowinnorm", models="linear", features="W",
                                  windows=[1.0], blocks=5, epochs=5, seed=0, no_plots=True, pairs=None,
                                  split=["T1,T2,T3:T4"], target="waist"))
        with open(os.path.join(out_dir + "_w", "log.txt"), encoding="utf-8") as f:
            good &= "正解そのもの" in f.read()
    except ro.RegError:
        good = False
    print(f"  {'OK ' if good else 'NG '} 加速度計 → 全身の ENMO（線形回帰の R²）: 腰 W {got[('linear', 'W')]['R2']:.3f}、"
          f"手首 R {got[('linear', 'R')]['R2']:.3f}、腰＋手首 W+R {got[('linear', 'W+R')]['R2']:.3f}、"
          f"ガウス過程 RS {got[('gpr', 'RS')]['R2']:.3f}")
    return good


def check_pca_pls(tmp):
    """Feature M (the Doppler map), --pca and PLS regression. / ドップラーの図そのもの・PCA・PLS 回帰"""
    import argparse
    inp = os.path.join(tmp, "in_pls")
    for k in range(4):
        make_trial(inp, k)
    res, out_dir = ro.run(argparse.Namespace(input=inp, out=os.path.join(tmp, "out_pls"), doppler="nowinnorm",
                                             models="linear,rf,pls", features="A,M,B", windows=[1.0], blocks=5,
                                             epochs=3, seed=0, no_plots=False, pairs=None, split=["T1,T2,T3:T4"],
                                             pca="auto", map_size="8x25"))
    got = {(r["model"], r["features"]): r for r in res}
    files = os.listdir(out_dir)
    with open(os.path.join(out_dir, "log.txt"), encoding="utf-8") as f:
        logtxt = f.read()
    keys = [("linear", "A PCA"), ("linear", "M PCA"), ("rf", "M PCA"), ("linear", "B"), ("pls", "A"), ("pls", "M")]
    good = (all(k in got for k in keys) and all(got[k]["R2"] > 0.8 for k in keys)
            and all(got[k].get("n_components") for k in keys if "PCA" in k[1] or k[0] == "pls")
            and "n_components" not in got[("linear", "B")]          # no PCA on the physics features / B にはかけない
            and "pls_coef.csv" in files and any(f.startswith("pls_coef_") and f.endswith("_M.png") for f in files)
            and "400 個" in logtxt and "PCA の成分数" in logtxt and "PLS の成分数" in logtxt)
    with open(os.path.join(out_dir, "pls_coef.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    good &= sum(1 for r in rows if r["features"] == "M") == 2 * 8 * 25
    try:
        ro.parse_pca("abc")
        good = False
    except ro.RegError:
        pass
    print(f"  {'OK ' if good else 'NG '} 特徴量 M・PCA・PLS: R² " + "、".join(f"{m} {fe} {got[(m, fe)]['R2']:.3f}"
                                                                    for m, fe in keys if (m, fe) in got)
          + f"、PLS の成分数 {got[('pls', 'M')].get('n_components')}")
    return good


def check_imu_cnn(tmp):
    """CNN on the accelerometer's 50 Hz ENMO waveform. / 加速度計の ENMO の波形の CNN"""
    import importlib.util
    if not importlib.util.find_spec("tensorflow"):
        print("  (TensorFlow が無いので加速度計の波形の CNN は飛ばします)")
        return True
    import argparse
    src = os.path.join(tmp, "in_waist")          # made by check_waist_input (body and wrist columns) / 全身・手首の列つき
    if not os.path.isdir(src):
        check_waist_input(tmp)
    good = True
    for part in ("waist", "wrist"):
        res, out_dir = ro.run(argparse.Namespace(input=src, out=os.path.join(tmp, f"out_imu_{part}"), doppler="nowinnorm",
                                                 models="cnn,cnn_imu,cnn_imu3s", features="B", windows=[1.0], blocks=5,
                                                 epochs=3, seed=0, no_plots=True, pairs=None, split=["T1,T2,T3:T4"],
                                                 target="body", imu_part=part, cnn_seeds=1, gain_jitter="3"))
        got = {r["model"]: r for r in res}
        with open(os.path.join(out_dir, "log.txt"), encoding="utf-8") as f:
            logtxt = f.read()
        ja = "腰" if part == "waist" else "手首"
        good &= (all(m in got for m in ("cnn", "cnn_imu", "cnn_imu3s"))
                 and got["cnn_imu"]["features"] == f"{ja} ENMO 波形 1秒" and got["cnn_imu3s"]["features"] == f"{ja} ENMO 波形 3秒"
                 and got["cnn_imu"].get("jitter", 0) == 0 and "強さ±3dB" in got["cnn"]["features"]
                 and "加速度計の波形の CNN" in logtxt and np.isfinite(got["cnn_imu3s"]["R2"]))
    trials = [ro.build_trial(sp, "nowinnorm", 1.0, [], need_c=False, column="ENMO_body(mG)",
                             need_imu=("ENMO(mg)" if False else ro.TARGET, [0.0, 1.0]))
              for sp in ro.discover(src, [])[:1]]
    good &= trials[0]["XI"][0.0].shape[1:] == (50, 1) and trials[0]["XI"][1.0].shape[1:] == (150, 1)
    print(f"  {'OK ' if good else 'NG '} 加速度計の波形の CNN: 腰・手首とも CNN と並ぶ、入力 {trials[0]['XI'][0.0].shape[1:]} / "
          f"{trials[0]['XI'][1.0].shape[1:]}")
    return good


def check_feature_de(tmp):
    """D and E run; with a different receive gain in each trial, E (shape only) beats A (absolute level).
    D・E が動くこと。試行ごとに受信の強さ（ゲイン）が違うとき、形だけの E が強さそのものの A より良いこと"""
    import argparse
    root = os.path.join(tmp, "in_de")
    v = (np.arange(N_FFT) - N_FFT // 2) * DV
    for k, gain in enumerate((0.3, 1.0, 3.0, 0.5, 2.0, 1.5)):
        name = f"G{k + 1}"
        rng = np.random.default_rng(70 + k)
        folder = os.path.join(root, name, ro.VARIANT_DIRS["nowinnorm"])
        os.makedirs(folder)
        t_frames = np.arange(0, 80.5, FRAME_DT)
        s_ = intensity(t_frames, k)
        swing = np.abs(np.sin(2 * np.pi * 0.9 * t_frames))
        for ant, g2 in (("antenna1", 1.0), ("antenna2", 0.6)):
            width = 0.15 + 0.6 * s_ * swing
            S = (np.exp(-0.5 * (v[None, :] / width[:, None]) ** 2) * s_[:, None] + 3.0 * (np.abs(v) < 1e-9)[None, :])
            S = gain * g2 * S * (1 + 0.1 * rng.standard_normal(S.shape)) ** 2 + 1e-4
            base = os.path.join(folder, f"{name}_{ant}_stream_0")
            with open(base + ".txt", "wb") as fp:
                pickle.dump(S, fp)
            np.save(base + "_time.npy", t_frames)
            with open(base + "_meta.json", "w") as fp:
                json.dump(dict(delta_v_bin=DV, n_fft=N_FFT, time_origin_s=0.0), fp)
        t50 = np.arange(0, 80.0, 0.02)
        en = 60 * intensity(t50, k) + rng.normal(0, 2.0, len(t50))
        with open(os.path.join(root, name, f"{name}_power_timeseries.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_csi(s)", "t_imu(s)", "ENMO(mG)"])
            for a, b in zip(t50, en):
                w.writerow([f"{a:.4f}", f"{a + 40:.4f}", f"{b:.3f}"])
    res, out_dir = ro.run(argparse.Namespace(input=root, out=os.path.join(tmp, "out_de"), doppler="nowinnorm",
                                             models="rf", features="A,D,E,A+D", windows=[1.0], blocks=5,
                                             epochs=5, seed=0, no_plots=True, pairs=None))
    r2 = {x["features"]: x["R2"] for x in res if x["model"] == "rf"}
    mae = {x["features"]: x["MAE"] for x in res if x["model"] == "rf"}
    with open(os.path.join(out_dir, "features_nowinnorm_1s.csv"), encoding="utf-8") as f:
        head = next(csv.reader(f))
    nd = sum(h.startswith("D_") for h in head)
    ne = sum(h.startswith("E_") for h in head)
    good = mae["E"] < 0.6 * mae["A"] and r2["E"] > r2["A"] and "A+D" in r2 and nd == 280 and ne == 200
    print(f"  {'OK ' if good else 'NG '} 特徴量 D・E: 試行ごとに受信の強さが違うとき、ランダムフォレストの R² "
          f"A {r2['A']:.3f}（MAE {mae['A']:.1f}）→ E {r2['E']:.3f}（MAE {mae['E']:.1f}）、D {r2['D']:.3f}、A+D {r2['A+D']:.3f}、"
          f"表の列 D {nd} 個・E {ne} 個")
    return good


def check_importance_context(tmp):
    """--importance and B+D@1 (features of the window before and after). / 重要度と前後の区間の特徴量"""
    import argparse
    inp = os.path.join(tmp, "in_imp")
    for k in range(4):
        make_trial(inp, k)
    out_dir = os.path.join(tmp, "out_imp")
    res, _ = ro.run(argparse.Namespace(input=inp, out=out_dir, doppler="nowinnorm", models="rf,xgboost",
                                       features="B+D,B+D@1", windows=[1.0], blocks=5, epochs=5, seed=0,
                                       no_plots=False, pairs=None, importance=True))
    files = os.listdir(out_dir)
    with open(os.path.join(out_dir, "importance.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    n_bd = sum(1 for r in rows if r["features"] == "B+D" and r["model"] == "XGBoost")
    n_ctx = sum(1 for r in rows if r["features"] == "B+D@1" and r["model"] == "XGBoost")
    has_t = any(r["feature"].startswith("t-1:") for r in rows) and any(r["feature"].startswith("t+1:") for r in rows)
    good = (n_bd == 300 and n_ctx == 900 and has_t and "importance_groups.csv" in files
            and any(f.startswith("importance_") and f.endswith(".png") for f in files)
            and all(r["R2"] > 0.5 for r in res if r["model"] != "baseline"))
    print(f"  {'OK ' if good else 'NG '} 特徴量の重要度と前後の区間: B+D {n_bd} 列、B+D@1 {n_ctx} 列（t-1・t+1 つき）、"
          f"図 {[f for f in files if f.startswith('importance_') and f.endswith('.png')]}")
    return good


def check_target_body(tmp):
    """--target body: the whole-body ENMO column (ENMO_body(mG)) is the ground truth; trials without it are left out.
    --target body: 全身の ENMO の列を正解にする。列がない試行（センサが 4 台そろっていない）は使わない"""
    import argparse
    inp = os.path.join(tmp, "in_body")
    for k in range(4):
        make_trial(inp, k)
    for k in range(3):   # T4 keeps the waist column only / T4 は腰の列だけ
        name = f"T{k + 1}"
        path = os.path.join(inp, name, "imu" if k == 3 else "", f"{name}_power_timeseries.csv")
        with open(path, encoding="utf-8") as f:
            rows = list(csv.reader(f))
        rng = np.random.default_rng(300 + k)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(rows[0] + ["ENMO_body(mG)"])
            for r in rows[1:]:
                w.writerow(r + [f"{300 * intensity(float(r[0]), k) + rng.normal(0, 8.0):.3f}"])
    out_dir = os.path.join(tmp, "out_body")
    res, _ = ro.run(argparse.Namespace(input=inp, out=out_dir, doppler="nowinnorm", models="linear", features="B",
                                       windows=[1.0], blocks=5, epochs=5, seed=0, no_plots=True, pairs=None,
                                       target="body"))
    r = [x for x in res if x["model"] == "linear"][0]
    names = sorted(str(n) for n in set(r["tname"]))
    with open(os.path.join(out_dir, "metrics.csv"), encoding="utf-8") as f:
        tg = {row["target"] for row in csv.DictReader(f)}
    with open(os.path.join(out_dir, "log.txt"), encoding="utf-8") as f:
        logtxt = f.read()
    good = (names == ["T1", "T2", "T3"] and r["R2"] > 0.5 and r["y"].mean() > 150 and tg == {"body"}
            and "全身の ENMO がない試行は使いません" in logtxt and "T4" in logtxt)
    print(f"  {'OK ' if good else 'NG '} 正解を全身の ENMO に（--target body）: 試行 {names}（T4 は列がないので除外）、"
          f"正解の平均 {r['y'].mean():.0f} mG、R² {r['R2']:.3f}")
    return good


def check_feature_f(tmp):
    """Feature F (Doppler summed over the moving bins -> time derivative -> 1 s mean of |d/dt|), the static-path
    check, and several derivative windows. / 特徴量 F・静止成分の広がりの確認・微分の窓を複数"""
    import argparse
    inp = os.path.join(tmp, "in_f")
    for k in range(4):
        make_trial(inp, k)
    out_dir = os.path.join(tmp, "out_f")
    res, _ = ro.run(argparse.Namespace(input=inp, out=out_dir, doppler="nowinnorm", models="linear",
                                       features="F1,F2a,F2b,F1+F2a+F2b", windows=[1.0], blocks=5, epochs=5, seed=0,
                                       no_plots=False, pairs=None, static_v=0.65, deriv_window="0.2"))
    r2 = {x["features"]: x["R2"] for x in res if x["model"] == "linear"}
    with open(os.path.join(out_dir, "features_nowinnorm_1s.csv"), encoding="utf-8") as f:
        head = next(csv.reader(f))
    fcols = [h for h in head if h.startswith("F")]
    files = os.listdir(out_dir)
    with open(os.path.join(out_dir, "log.txt"), encoding="utf-8") as f:
        logtxt = f.read()
    res2, _ = ro.run(argparse.Namespace(input=inp, out=os.path.join(tmp, "out_f2"), doppler="nowinnorm",
                                        models="linear", features="F2b", windows=[1.0], blocks=5, epochs=5, seed=0,
                                        no_plots=False, pairs=None, deriv_window="0.1,0.4"))
    labels = sorted({x["features"] for x in res2 if x["model"] == "linear"})
    good = (fcols == ["F1_ant1", "F1_ant2", "F2A_ant1", "F2A_ant2", "F2B_ant1", "F2B_ant2"]
            and r2["F2B"] > 0.5 and r2["F1+F2A+F2B"] > 0.5
            and "static_spectrum_nowinnorm_1s.csv" in files and "static_spectrum_nowinnorm_1s.png" in files
            and "静止成分の広がり" in logtxt and labels == ["F2B/0.1s", "F2B/0.4s"])
    print(f"  {'OK ' if good else 'NG '} 特徴量 F: 線形回帰の R² " + "、".join(f"{k} {v:.3f}" for k, v in r2.items())
          + f"、列 {fcols}、微分の窓を 2 つ → {labels}、静止成分の図 "
          + str([f for f in files if f.startswith("static_spectrum")]))
    return good


def check_similar_names(tmp):
    """A truth whose Doppler is missing is not paired with a similarly named one (A3_XL1 vs A3_XR1).
    ドップラーがない正解データを、似た名前の別の試行（A3_XL1 と A3_XR1）と組にしない"""
    import argparse
    src = os.path.join(tmp, "src_sim")
    for k in range(3):
        make_trial(src, k)
    root = os.path.join(tmp, "in_sim")
    names = {0: ("A3_XL2", "A3_XL2"), 1: ("A3_XL1", "A3_XR1"), 2: ("A3_XL3", "A3_XL3")}   # T2's Doppler is XR1
    for k, (truth, dop) in names.items():
        tn = f"T{k + 1}"
        tdir = os.path.join(root, "A3", "imu", truth)
        os.makedirs(tdir, exist_ok=True)
        ts = os.path.join(src, tn, "imu", f"{tn}_power_timeseries.csv")
        shutil.copy(ts if os.path.exists(ts) else os.path.join(src, tn, f"{tn}_power_timeseries.csv"),
                    os.path.join(tdir, f"{truth}_power_timeseries.csv"))
        folder = ro.VARIANT_DIRS["nowinnorm"]
        os.makedirs(os.path.join(root, "A3", folder), exist_ok=True)
        for f in os.listdir(os.path.join(src, tn, folder)):
            shutil.copy(os.path.join(src, tn, folder, f), os.path.join(root, "A3", folder, f.replace(tn, dop)))
    res, out_dir = ro.run(argparse.Namespace(input=root, out=os.path.join(tmp, "out_sim"), doppler="nowinnorm",
                                             models="linear", features="F2b", windows=[1.0], blocks=5, epochs=5,
                                             seed=0, no_plots=True, pairs=None))
    used = sorted(str(n) for n in set(res[0]["tname"]))
    with open(os.path.join(out_dir, "log.txt"), encoding="utf-8") as f:
        logtxt = f.read()
    good = used == ["A3_XL2", "A3_XL3"] and "ドップラーが見つからない正解データ: A3_XL1" in logtxt
    print(f"  {'OK ' if good else 'NG '} 似た名前の別の試行と組にしない: 使った試行 {used}（A3_XL1 は A3_XR1 と組にせず飛ばす）")
    return good


def check_levels(tmp):
    """--levels_only: the per-trial level table shows each trial's receive gain (static part) and exits.
    試行ごとの強さの表: 試行ごとの受信の強さ（静止成分）が出て、学習せずに終わる"""
    import argparse
    root = os.path.join(tmp, "in_lv")
    v = (np.arange(N_FFT) - N_FFT // 2) * DV
    gains = (1.0, 2.0, 0.5)
    for k, gain in enumerate(gains):
        name = f"A{k + 1}_XL1"
        folder = os.path.join(root, name, ro.VARIANT_DIRS["nowinnorm"])
        os.makedirs(folder)
        t_frames = np.arange(0, 80.5, FRAME_DT)
        s_ = intensity(t_frames, k)
        for ant in ("antenna1", "antenna2"):
            S = gain * (np.exp(-0.5 * (v[None, :] / (0.15 + 0.6 * s_[:, None])) ** 2) * s_[:, None]
                        + 3.0 * (np.abs(v) < 1e-9)[None, :]) + 1e-4
            base = os.path.join(folder, f"{name}_{ant}_stream_0")
            with open(base + ".txt", "wb") as fp:
                pickle.dump(S, fp)
            np.save(base + "_time.npy", t_frames)
            with open(base + "_meta.json", "w") as fp:
                json.dump(dict(delta_v_bin=DV, n_fft=N_FFT, time_origin_s=0.0), fp)
        t50 = np.arange(0, 80.0, 0.02)
        with open(os.path.join(root, name, f"{name}_power_timeseries.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_csi(s)", "t_imu(s)", "ENMO(mG)"])
            for a in t50:
                w.writerow([f"{a:.4f}", f"{a + 40:.4f}", f"{60 * intensity(a, k):.3f}"])
    out_dir = os.path.join(tmp, "out_lv")
    res, _ = ro.run(argparse.Namespace(input=root, out=out_dir, doppler="nowinnorm", models="linear,cnn",
                                       features="F2b", windows=[1.0], blocks=5, epochs=5, seed=0, no_plots=True,
                                       pairs=None, levels_only=True))
    with open(os.path.join(out_dir, "trial_levels.csv"), encoding="utf-8") as f:
        rows = {r["trial"]: r for r in csv.DictReader(f)}
    st = {k: float(r["ant1_static(dB)"]) for k, r in rows.items()}
    want = {f"A{k + 1}_XL1": 10 * np.log10(g) for k, g in enumerate(gains)}
    good = (res == [] and all(abs(st[k] - want[k]) < 0.6 for k in want)   # slow motion also falls in the static bins / 遅い動きも静止ビンに入る分
            and not any(f.startswith("metrics") for f in os.listdir(out_dir)))
    print(f"  {'OK ' if good else 'NG '} 試行ごとの強さの表（--levels_only）: 静止成分 "
          + "、".join(f"{k} {st[k]:+.1f} dB（受信の強さ {want[k]:+.1f} dB）" for k in sorted(st)) + "、学習はしない")
    return good


def check_two_datasets(tmp):
    """regression_input/A1 and A2 side by side, and --split. / A1 と A2 を並べて置き、学習とテストを指定する"""
    import argparse
    src = os.path.join(tmp, "src_two")
    for k in range(8):
        make_trial(src, k)
    root = os.path.join(tmp, "in_two")
    trials = {"A1": [0, 1, 2, 3], "A2": [4, 5, 6, 7]}
    for ds, ks in trials.items():
        for j, k in enumerate(ks):
            tn, trial = f"T{k + 1}", f"{ds}_XR{j + 1}"
            tdir = os.path.join(root, ds, "imu", trial)
            os.makedirs(tdir, exist_ok=True)
            ts = os.path.join(src, tn, "imu", f"{tn}_power_timeseries.csv")
            shutil.copy(ts if os.path.exists(ts) else os.path.join(src, tn, f"{tn}_power_timeseries.csv"),
                        os.path.join(tdir, f"{trial}_power_timeseries.csv"))
            if trial == "A2_XR3":
                continue  # no Doppler for A2_XR3: it must not take A1_XR3's / A2_XR3 はドップラーなし（A1_XR3 と組にしない）
            folder = ro.VARIANT_DIRS["nowinnorm"]
            os.makedirs(os.path.join(root, ds, folder), exist_ok=True)
            for f in os.listdir(os.path.join(src, tn, folder)):
                shutil.copy(os.path.join(src, tn, folder, f),
                            os.path.join(root, ds, folder, f.replace(f"{tn}_antenna", f"{trial}_antenna")))
    # a complete trial in "その他": never read / 「その他」の中の試行は読まない
    oth = os.path.join(root, "その他")
    os.makedirs(os.path.join(oth, "imu", "A2_XHR1"), exist_ok=True)
    shutil.copy(os.path.join(root, "A1", "imu", "A1_XR1", "A1_XR1_power_timeseries.csv"),
                os.path.join(oth, "imu", "A2_XHR1", "A2_XHR1_power_timeseries.csv"))
    shutil.copytree(os.path.join(root, "A1", ro.VARIANT_DIRS["nowinnorm"]), os.path.join(oth, ro.VARIANT_DIRS["nowinnorm"]))
    for f in os.listdir(os.path.join(oth, ro.VARIANT_DIRS["nowinnorm"])):
        os.rename(os.path.join(oth, ro.VARIANT_DIRS["nowinnorm"], f),
                  os.path.join(oth, ro.VARIANT_DIRS["nowinnorm"], f.replace("A1_XR", "A2_XHR")))
    out_dir = os.path.join(tmp, "out_two")
    res, _ = ro.run(argparse.Namespace(input=root, out=out_dir, doppler="nowinnorm", models="linear", features="B",
                                       windows=[1.0], blocks=5, epochs=5, seed=0, no_plots=False, pairs=None,
                                       split=["A1:A2", "A1,A2_XR1:A2_XR2", "cv"]))
    by = {r["split"]: r for r in res if r["model"] == "linear"}
    s1, s2 = "学習 A1 → テスト A2", "学習 A1+A2_XR1 → テスト A2_XR2"
    names_all = sorted(set(by["試行ごと"]["tname"]))
    good = (set(by) == {s1, s2, "試行ごと"}
            and sorted(set(by[s1]["tname"])) == ["A2_XR1", "A2_XR2", "A2_XR4"]
            and sorted(set(by[s2]["tname"])) == ["A2_XR2"] and len(by[s2]["y"]) == 80
            and "A2_XR3" not in names_all and len(names_all) == 7 and not any("XHR" in n for n in names_all)
            and all(by[k]["R2"] > 0.5 for k in by))
    files = os.listdir(out_dir)
    good &= any(f.startswith("pred_train_A1_test_A2_") for f in files)
    with open(os.path.join(out_dir, "metrics.csv"), encoding="utf-8") as f:
        good &= next(csv.reader(f))[2] == "split"
    try:  # a name that does not exist / ない名前
        ro.run(argparse.Namespace(input=root, out=out_dir, doppler="nowinnorm", models="linear", features="B",
                                  windows=[1.0], blocks=5, epochs=5, seed=0, no_plots=True, pairs=None,
                                  split=["A1:A3"]))
        good = False
    except ro.RegError as e:
        good &= "A3" in str(e)
    print(f"  {'OK ' if good else 'NG '} A1 と A2 を並べた置き方と --split: 試行 {names_all}（A2_XR3 はドップラーなしで飛ばす）、"
          + "、".join(f"{k}: R² {v['R2']:.3f}" for k, v in by.items()))
    return good


def dataset_layout(tmp, related_names=True):
    """The layout of a dataset folder (everything of A1 together). / データセットのフォルダにまとめた置き方
        <in>/A1/imu/<trial>/imu/<trial>_power_timeseries.csv
        <in>/A1/doppler_traces_nowinnorm/<CSI name>_antenna{1,2}_..._stream_0.txt  (all trials)
    With related_names=False the Doppler names do not contain the trial name, and a batch_pairs_*.csv
    (from the calculator's batch run) gives the pairs. / 名前が対応しないときは一括処理の組み合わせの一覧で"""
    import csv as _csv
    src = os.path.join(tmp, "src" + ("" if related_names else "_pairs"))
    for k in range(5):
        make_trial(src, k)
    root = os.path.join(tmp, "in_ds" + ("" if related_names else "_pairs"))
    ds = os.path.join(root, "A1")
    rows = []
    for k in range(5):
        tn, trial = f"T{k + 1}", f"A1_XR{k + 1}"
        tdir = os.path.join(ds, "imu", trial, "imu")
        os.makedirs(tdir, exist_ok=True)
        shutil.copy(os.path.join(src, tn, "imu", f"{tn}_power_timeseries.csv") if os.path.exists(
            os.path.join(src, tn, "imu", f"{tn}_power_timeseries.csv")) else os.path.join(src, tn, f"{tn}_power_timeseries.csv"),
            os.path.join(tdir, f"{trial}_power_timeseries.csv"))
        csi = trial if related_names else f"A1_walk{(k * 3) % 5 + 1}"
        for folder in ro.VARIANT_DIRS.values():
            os.makedirs(os.path.join(ds, folder), exist_ok=True)
            for f in os.listdir(os.path.join(src, tn, folder)):
                shutil.copy(os.path.join(src, tn, folder, f),
                            os.path.join(ds, folder, f.replace(f"{tn}_antenna", f"{csi}_antenna").replace(
                                "_stream_0", "_WAX202_aligned_CSI_stream_0")))
        rows.append([f"{trial}.csv", f"{csi}_antenna1_WAX202_aligned_CSI.mat"])
    if not related_names:
        with open(os.path.join(root, "batch_pairs_20270101_000000.csv"), "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["use", "imu", "csi", "how"])
            for r in rows:
                w.writerow(["1"] + r + ["時刻"])
    return root


def main():
    tmp = tempfile.mkdtemp(prefix="regress_")
    ok = True
    try:
        inp = os.path.join(tmp, "in")
        for k in range(5):
            make_trial(inp, k, old_meta=(k == 4))
        # the dataset-folder layout (quick, linear model only) / データセットのフォルダの置き方（線形回帰だけで確認）
        import argparse
        for related in (True, False):
            root = dataset_layout(tmp, related)
            res, _ = ro.run(argparse.Namespace(input=root, out=os.path.join(tmp, "out_ds"), doppler="nowinnorm",
                                               models="linear", features="B", windows=[1.0], blocks=5, epochs=5,
                                               seed=0, no_plots=True, pairs=None))
            r = [x for x in res if x["model"] == "linear"][0]
            names = sorted(set(r["tname"]))
            good = names == [f"A1_XR{k}" for k in range(1, 6)] and r["R2"] > 0.5
            label = "名前で対応" if related else "組み合わせの一覧で対応"
            print(f"  {'OK ' if good else 'NG '} データセットのフォルダの置き方（{label}）: {names}, R² {r['R2']:.3f}")
            ok &= good
        ok &= check_feature_c(tmp)
        ok &= check_two_datasets(tmp)
        ok &= check_feature_de(tmp)
        ok &= check_importance_context(tmp)
        ok &= check_target_body(tmp)
        ok &= check_feature_f(tmp)
        ok &= check_similar_names(tmp)
        ok &= check_levels(tmp)
        ok &= check_cnn_study(tmp)
        ok &= check_gain_jitter(tmp)
        ok &= check_mape(tmp)
        ok &= check_waist_input(tmp)
        ok &= check_pca_pls(tmp)
        ok &= check_imu_cnn(tmp)
        import importlib.util
        if importlib.util.find_spec("tensorflow"):
            models = "linear,rf,xgboost,mlp,svr,gpr,cnn"
        else:
            models = "linear,rf,xgboost,mlp,svr,gpr"
            print("(TensorFlow が無いので CNN は飛ばします)")
        out_dir = os.path.join(tmp, "out")
        results, _ = ro.run(ro_args(inp, out_dir, models))
        got = {(r["variant"], r["window"], r["model"], r["features"]): r for r in results}
        W = (1.0, 5.0)
        files = sorted(os.listdir(out_dir))
        checks = {
            '3 つの版 x 2 つの区間 x (基準 + モデル) が全部ある':
                all((v, w, "baseline", "-") in got for v in ro.VARIANT_DIRS for w in W)
                and all((v, w, m, f) in got for v in ro.VARIANT_DIRS for w in W
                        for m in ("linear", "rf", "xgboost", "mlp") for f in ("A", "B", "C")),
            '基準 (平均値) の R² は 0 前後以下':
                all(got[(v, w, "baseline", "-")]["R2"] < 0.1 for v in ro.VARIANT_DIRS for w in W),
            '線形回帰 (特徴量 B, nowinnorm) の R² > 0.5 (1 秒・5 秒)':
                all(got[("nowinnorm", w, "linear", "B")]["R2"] > 0.5 for w in W),
            'ランダムフォレスト (特徴量 B, nowinnorm) の R² > 0.5 (1 秒・5 秒)':
                all(got[("nowinnorm", w, "rf", "B")]["R2"] > 0.5 for w in W),
            'SVR とガウス過程回帰 (特徴量 B, nowinnorm) の R² > 0.5 (1 秒・5 秒)':
                all(got[("nowinnorm", w, m, "B")]["R2"] > 0.5 for w in W for m in ("svr", "gpr")),
            '表と図ができている': all(f in files for f in ("metrics.csv", "metrics_by_trial.csv", "predictions.csv",
                                                        "comparison.png", "log.txt"))
                                and any(f.startswith("pred_nowinnorm_1s_linear") for f in files)
                                and any(f.startswith("pred_nowinnorm_5s_linear") for f in files),
        }
        if "cnn" in models:
            checks['CNN の結果がある'] = all((v, w, "cnn", "Doppler 2D") in got for v in ro.VARIANT_DIRS for w in W)
        print("\n結果:")
        for k, v in checks.items():
            print(f"  {'OK ' if v else 'NG '} {k}")
            ok &= bool(v)
        # alignment check: the window grid should start at 1 s (data from 0.06 s) / 区間の並び
        r = got[("nowinnorm", 1.0, "linear", "B")]
        first = r["win"][r["tname"] == "T1"][0]
        n1 = int(np.sum(r["tname"] == "T1"))
        al = abs(first[0]) < 1e-9 and n1 == 80
        print(f"  {'OK ' if al else 'NG '} 区間は加速度計算ソフトと同じ 0, 1, 2 … 秒の区切りで 80 区間 "
              f"(T1 の最初 {first[0]:.2f}〜{first[1]:.2f} s、{n1} 区間)")
        ok &= al
        n5 = int(np.sum(got[("nowinnorm", 5.0, "linear", "B")]["tname"] == "T1"))
        print(f"  {'OK ' if n5 == 16 else 'NG '} 5 秒の区間は 16 個 (T1: {n5} 個)")
        ok &= n5 == 16
    finally:
        if "--keep" in sys.argv:
            print(f"(kept {tmp})")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    print("\nすべてのテストに合格しました。" if ok else "\n不合格があります。")
    return 0 if ok else 1


def ro_args(inp, out_dir, models):
    import argparse
    return argparse.Namespace(input=inp, out=out_dir, doppler="all", models=models, features="A,B,C",
                              windows=[1.0, 5.0], blocks=5, epochs=40, seed=0, no_plots=False)


if __name__ == "__main__":
    sys.exit(main())
