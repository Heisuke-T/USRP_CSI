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
    os.makedirs(os.path.join(d, "imu"))
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
    with open(os.path.join(d, "imu", f"{name}_power_timeseries.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_csi(s)", "t_imu(s)", "ENMO(mg)"])
        for a, b in zip(t50, en):
            w.writerow([f"{a:.4f}", f"{a + 40:.4f}", f"{b:.3f}"])


def main():
    tmp = tempfile.mkdtemp(prefix="regress_")
    ok = True
    try:
        inp = os.path.join(tmp, "in")
        for k in range(5):
            make_trial(inp, k, old_meta=(k == 4))
        import importlib.util
        if importlib.util.find_spec("tensorflow"):
            models = "linear,rf,xgboost,mlp,cnn"
        else:
            models = "linear,rf,xgboost,mlp"
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
                        for m in ("linear", "rf", "xgboost", "mlp") for f in ("A", "B")),
            '基準 (平均値) の R² は 0 前後以下':
                all(got[(v, w, "baseline", "-")]["R2"] < 0.1 for v in ro.VARIANT_DIRS for w in W),
            '線形回帰 (特徴量 B, nowinnorm) の R² > 0.5 (1 秒・5 秒)':
                all(got[("nowinnorm", w, "linear", "B")]["R2"] > 0.5 for w in W),
            'ランダムフォレスト (特徴量 B, nowinnorm) の R² > 0.5 (1 秒・5 秒)':
                all(got[("nowinnorm", w, "rf", "B")]["R2"] > 0.5 for w in W),
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
    return argparse.Namespace(input=inp, out=out_dir, doppler="all", models=models, features="A,B",
                              windows=[1.0, 5.0], blocks=5, epochs=40, seed=0, no_plots=False)


if __name__ == "__main__":
    sys.exit(main())
