"""
    regress_enmo.py の検証 (合成データ)。

    5 試行ぶんの入力フォルダを作る。各試行の中で動きの強さ s(t) が変わり (試行ごとに違う運動)、
      ドップラー: 足踏みの周期で速さが揺れる反射。速度の広がりと強さが s(t) に比例
                  (アンテナ 2 本、時刻は alignCSI の時間軸、_meta.json に time_origin_s)
      IMU:       ENMO = 60 * s(t) + 雑音 [mg] (加速度計算ソフトと同じ列の _power_timeseries.csv)
    これで regress_enmo.py を 1 秒と 5 秒の区間で回し、
      1. 3 つのドップラーの版・5 モデル + 基準 (平均値) が最後まで動き、表と図ができること
      2. 物理特徴量 B の線形回帰などが、平均値で予測する基準よりはっきり良いこと (R² > 0.5)
      3. 区間が加速度計算ソフトと同じ 0, 1, 2 … 秒の区切りになり、古い形式の meta (time_origin_s なし) でも動くこと
      4. 特徴量 C (速さの時間微分): 足踏みのテンポが試行ごとに違い、ENMO = 速さ x テンポ (加速度) のとき、
         線形回帰で B に C を足すと はっきり良くなること (速さだけでは当たらない分を C が拾う)
      6. CNN の確認 (--cnn_study, --cnn_seeds): 理由を確かめる CNN 4 種・乱数の種ごとの結果・Grad-CAM の図ができること
      7. SVR とガウス過程回帰 (特徴量 B) も基準よりはっきり良いこと
      5. regression_input/A1 と A2 を並べて置いたとき: データセットをまたいで名前で組にしないこと、
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
        w.writerow(["t_csi(s)", "t_imu(s)", "ENMO(mg)"])
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
            w.writerow(["t_csi(s)", "t_imu(s)", "ENMO(mg)"])
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
            and "A2_XR3" not in names_all and len(names_all) == 7
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
        ok &= check_cnn_study(tmp)
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
