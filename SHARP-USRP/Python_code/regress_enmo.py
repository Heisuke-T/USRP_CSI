"""
    ドップラーから腰の加速度の指標 (ENMO) を推定する回帰モデルを作り、比べる。

    正解: 加速度計算ソフト (witmotion_wt011 の compute_power.py) が出す腰の ENMO
          (max(合成加速度 - 1 g, 0) [mg]) の区間 (既定 1 秒と 5 秒) ごとの平均
    入力: SHARP-USRP のドップラー (doppler_traces*/ の数値の配列。PNG ではない)

    モデル (5 種類):
      線形回帰 (Ridge)    : 特徴量 A / B
      ランダムフォレスト  : 特徴量 A / B
      XGBoost             : 特徴量 A / B
      MLP                 : 特徴量 A / B
      CNN                 : 1 秒分のドップラーの 2 次元配列 (時間 x 速度 x アンテナ 2 本)
    特徴量 A = 1 秒の中で速度ビンごとに平均したスペクトル (log10, アンテナ 2 本で 200 個)
    特徴量 B = 物理に基づく特徴量 (速さの平均・2 乗平均・広がり・最大速度・足踏みの周期など,
               アンテナごと)
    どのモデルも出力は区間ごとの ENMO の値 (数値)。最後に予測と正解を図にする。

    評価: 試行ごとに分ける (1 試行をテスト、残りで学習、を全試行について繰り返す)。
          隣り合う 1 秒どうしはよく似ているので、ランダムに分けると精度が高く出すぎるため。
          試行が 1 本しかないときは、時間を 5 つのかたまりに分けて同じことをする。

    入力フォルダ (regression_input/ の README.md を参照):
      regression_input/<試行名>/
          <名前>_power_timeseries.csv  (正解データ。加速度計算ソフトの出力。imu/ の中に置いてもよい)
          doppler_traces/            *antenna1*.txt, *antenna2*.txt (+ _time.npy, _meta.json)
          doppler_traces_nowinnorm/  同上 (--save_unnormalized の版)
          doppler_traces_raw/        同上

    例:
      python regress_enmo.py                        # nowinnorm 版で 5 モデル、1 秒と 5 秒の区間
      python regress_enmo.py --doppler all          # 3 つの版それぞれで比べる
      python regress_enmo.py --windows 1            # 1 秒の区間だけ
      python regress_enmo.py --models linear,rf     # 一部のモデルだけ

    Copyright (C) 2026 Heisuke Takeda
    Released under the GNU GPL v3 (see LICENSE).
"""

import argparse
import csv
import glob
import json
import os
import pickle
import sys
import time
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VARIANT_DIRS = {"normalized": "doppler_traces", "nowinnorm": "doppler_traces_nowinnorm",
                "raw": "doppler_traces_raw"}
MODEL_NAMES = {"linear": "線形回帰 (Ridge)", "rf": "ランダムフォレスト", "xgboost": "XGBoost",
               "mlp": "MLP", "cnn": "CNN"}
ALL_MODELS = ["linear", "rf", "xgboost", "mlp", "cnn"]
CNN_FRAMES = 64          # CNN input frames per second of window (at most 320) / CNN に入れる 1 秒あたりの時間フレーム数（最大 320）
TARGET = "ENMO(mg)"      # target column in the work-rate software's output / 正解の列
EPS = 1e-12


class RegError(Exception):
    """An error to show to the user. / 利用者に見せるエラー"""


def log(msg, out):
    print(msg, flush=True)
    out.append(msg)


# ============================================================ loading
def _one(pattern, what):
    hits = sorted(glob.glob(pattern))
    hits = [h for h in hits if not h.endswith(("_time.npy", "_meta.json"))]
    if not hits:
        raise RegError(f"{what} が見つかりません ({pattern})")
    if len(hits) > 1:
        raise RegError(f"{what} が複数あります。1 つにしてください: {', '.join(os.path.basename(h) for h in hits)}")
    return hits[0]


def load_doppler(folder, antenna, out):
    """One antenna's Doppler: (frames x bins) array, frame times on the CSI axis [s], velocity axis [m/s].
    1 本のアンテナのドップラー: (時間フレーム x 速度ビン) の配列、各フレームの CSI の時刻、速度軸"""
    txt = _one(os.path.join(folder, f"*{antenna}*.txt"), f"{os.path.basename(folder)} の {antenna} のドップラー")
    with open(txt, "rb") as fp:
        arr = np.asarray(pickle.load(fp), dtype=float)
    base = txt[:-4]
    if not os.path.exists(base + "_time.npy"):
        raise RegError(f"{os.path.basename(base)}_time.npy がありません (ドップラーと一緒にコピーしてください)")
    t = np.load(base + "_time.npy").astype(float)
    meta = {}
    if os.path.exists(base + "_meta.json"):
        with open(base + "_meta.json", encoding="utf-8") as fp:
            meta = json.load(fp)
    origin = meta.get("time_origin_s")
    if origin is None:
        # older Doppler files: the first packet is about start x Tc after the aligned file's 0
        # 古いドップラー: 最初のパケットは先頭から start 個 x Tc あたり
        origin = float(meta.get("start", 10)) * float(meta.get("Tc") or 0.006)
        log(f"    注意: {os.path.basename(base)}_meta.json に time_origin_s がないので、時刻を "
            f"{origin * 1e3:.0f} ms ずらして見積もります (数十 ms の誤差。ドップラーを計算し直すと正確になります)", out)
    n = min(len(t), len(arr))
    arr, t = arr[:n], t[:n] + origin
    n_fft = arr.shape[1]
    dv = float(meta.get("delta_v_bin") or 1.0)
    v = (np.arange(n_fft) - n_fft // 2) * dv
    return dict(S=arr, t=t, v=v, meta=meta, file=os.path.basename(txt))


def load_target(trial_dir, window):
    """ENMO per sample (t_csi, value), or per window. / ENMO（各時刻、または区間ごと）"""
    # directly in <trial>/ (or in <trial>/imu/ as before) / <試行>/ の直下（前の置き方の <試行>/imu/ でもよい）
    ts, win = [], []
    for sub in ("", "imu"):
        ts = ts or sorted(glob.glob(os.path.join(trial_dir, sub, "*_power_timeseries.csv")))
        win = win or sorted(glob.glob(os.path.join(trial_dir, sub, f"*_windows_{window:g}s.csv")))
    if ts:
        path, kind = ts[0], "timeseries"
    elif win:
        path, kind = win[0], "windows"
    else:
        raise RegError(f"{trial_dir}/ (または {trial_dir}/imu/) に *_power_timeseries.csv "
                       f"(または *_windows_{window:g}s.csv) がありません")
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or TARGET not in rows[0]:
        raise RegError(f"{os.path.basename(path)} に ENMO の列 ({TARGET}) がありません。"
                       "加速度計算ソフトを最新にして計算し直してください")
    if kind == "timeseries":
        t = np.array([float(r["t_csi(s)"]) for r in rows])
        y = np.array([float(r[TARGET]) for r in rows])
        return dict(kind=kind, t=t, y=y, file=os.path.basename(path))
    a = np.array([float(r["start_csi(s)"]) for r in rows])
    b = np.array([float(r["end_csi(s)"]) for r in rows])
    y = np.array([float(r[TARGET]) for r in rows])
    return dict(kind=kind, a=a, b=b, y=y, file=os.path.basename(path))


# ============================================================ features
def physics_features(S, v, t, t_ctx=None, S_ctx=None):
    """Feature set B for one antenna and one window. / 特徴量 B（アンテナ 1 本・1 区間）

    S: frames x bins (power), v: velocity of each bin.
    The step cadence needs a longer view, so it uses S_ctx/t_ctx (at least 5 s around the window).
    足踏みの周期は 1 秒では測れないので、区間を中心に 5 秒以上（S_ctx）から求める
    """
    P = np.clip(S, 0, None)
    av = np.abs(v)
    move = av > 1.5 * np.min(av[av > 0]) if np.any(av > 0) else np.ones_like(av, bool)  # leave out the 0 m/s bins
    tot = P.sum(axis=1) + EPS
    mov = P[:, move].sum(axis=1) + EPS
    w = P[:, move] / mov[:, None]
    m1 = (w * av[move]).sum(axis=1)                 # mean speed / 平均の速さ
    m2 = (w * av[move] ** 2).sum(axis=1)            # mean squared speed / 速さの 2 乗の平均
    spread = np.sqrt(np.clip(m2 - m1 ** 2, 0, None))
    cs = np.cumsum(w[:, np.argsort(av[move])], axis=1)
    v_sorted = np.sort(av[move])
    v95 = v_sorted[np.minimum(np.argmax(cs >= 0.95, axis=1), len(v_sorted) - 1)]
    cad = 0.0
    if S_ctx is not None and len(S_ctx) > 16 and t_ctx is not None and np.ptp(t_ctx) > 1.5:
        Pc = np.clip(S_ctx, 0, None)
        wc = Pc[:, move] / (Pc[:, move].sum(axis=1, keepdims=True) + EPS)
        sp = (wc * av[move]).sum(axis=1)
        tu = np.linspace(t_ctx[0], t_ctx[-1], 256)
        x = np.interp(tu, t_ctx, sp)
        x = x - x.mean()
        f = np.fft.rfftfreq(len(tu), tu[1] - tu[0])
        A = np.abs(np.fft.rfft(x * np.hanning(len(x))))
        band = (f >= 0.5) & (f <= 4.0)
        if band.any() and A[band].max() > 0:
            cad = float(f[band][np.argmax(A[band])])
    return np.array([
        np.log10(np.mean(tot)),                     # total power / 全体の強さ
        np.log10(np.mean(mov)),                     # moving power / 動きの成分の強さ
        np.mean(mov / tot),                         # moving / total
        np.mean(m1), np.std(m1),                    # mean speed and its variation / 平均の速さとその変動
        np.mean(m2),                                # mean squared speed (kinetic-energy-like) / 運動エネルギーに近い量
        np.mean(spread),                            # spread / 速度の広がり
        np.mean(v95),                               # 95 % speed / 最大に近い速さ
        cad,                                        # step cadence [Hz] / 足踏みの周期
        np.mean(m2) * cad,                          # power-like / 仕事率に近い量
    ])


B_NAMES = ["log_total", "log_moving", "moving_ratio", "mean_speed", "std_speed", "mean_speed2",
           "spread", "v95", "cadence", "speed2_x_cadence"]


def cnn_input(S, t, a, b):
    """Window's Doppler resampled to a fixed number of frames (log10). / CNN の入力（時間を一定数にそろえる）"""
    n = int(min(320, round(CNN_FRAMES * (b - a))))
    tu = a + (np.arange(n) + 0.5) * (b - a) / n
    out = np.empty((n, S.shape[1]))
    for j in range(S.shape[1]):
        out[:, j] = np.interp(tu, t, S[:, j])
    return np.log10(np.clip(out, EPS, None))


def build_trial(trial_dir, variant, window, out):
    """Windows of one trial: features A, B, CNN input and the target. / 1 試行分の区間"""
    name = os.path.basename(trial_dir.rstrip("/\\"))
    dop_dir = os.path.join(trial_dir, VARIANT_DIRS[variant])
    if not os.path.isdir(dop_dir):
        raise RegError(f"{name}: {VARIANT_DIRS[variant]}/ がありません")
    ants = [load_doppler(dop_dir, a, out) for a in ("antenna1", "antenna2")]
    tgt = load_target(trial_dir, window)
    if tgt["kind"] == "timeseries":
        t0 = max(0.0, max(d["t"][0] for d in ants), tgt["t"][0])
        t1 = min(min(d["t"][-1] for d in ants), tgt["t"][-1])
        # windows on the same grid as the power software (0, 1, 2 ... s); allow 0.1 s missing at the ends
        # 加速度計算ソフトと同じ区切り（0, 1, 2 … 秒）。端の 0.1 秒までの欠けは許す
        edges = np.arange(np.floor(t0 / window) * window, t1 - window + 0.1 + 1e-9, window)
        edges = edges[edges >= t0 - 0.1 - 1e-9]
        wins = [(e, e + window) for e in edges]
    else:
        wins = [(a, b) for a, b in zip(tgt["a"], tgt["b"])]
    XA, XB, XC, y, kept = [], [], [], [], []
    rate = np.median([1 / np.median(np.diff(d["t"])) for d in ants])
    dropped = 0
    for i, (a, b) in enumerate(wins):
        if tgt["kind"] == "timeseries":
            m = (tgt["t"] >= a) & (tgt["t"] < b)
            if m.sum() < 3:
                dropped += 1
                continue
            yv = float(np.mean(tgt["y"][m]))
        else:
            yv = float(tgt["y"][i])
        fa, fb, fc = [], [], []
        ok = True
        for d in ants:
            sel = (d["t"] >= a) & (d["t"] < b)
            if sel.sum() < 0.5 * rate * (b - a):  # too few frames (packet gap) / フレームが少なすぎる
                ok = False
                break
            half = max(2.5, (b - a) / 2)
            ctx = (d["t"] >= (a + b) / 2 - half) & (d["t"] < (a + b) / 2 + half)
            S = d["S"][sel]
            fa.append(np.log10(np.clip(S.mean(axis=0), EPS, None)))
            fb.append(physics_features(S, d["v"], d["t"][sel], d["t"][ctx], d["S"][ctx]))
            fc.append(cnn_input(d["S"], d["t"], a, b))
        if not ok:
            dropped += 1
            continue
        XA.append(np.concatenate(fa))
        XB.append(np.concatenate(fb))
        XC.append(np.stack(fc, axis=-1))
        y.append(yv)
        kept.append((a, b))
    if not y:
        raise RegError(f"{name}: ドップラーと IMU が重なる区間がありません (時間軸を確認してください)")
    log(f"  {name}: {len(y)} 区間（{window:g} 秒ごと、CSI の {kept[0][0]:.1f}〜{kept[-1][1]:.1f} s）"
        + (f"、{dropped} 区間はデータ不足で除外" if dropped else "")
        + f"  [{ants[0]['file']}, {ants[1]['file']}, {tgt['file']}]", out)
    return dict(name=name, XA=np.array(XA), XB=np.array(XB), XC=np.array(XC), y=np.array(y),
                win=np.array(kept), v=ants[0]["v"])


# ============================================================ models
def make_tabular(kind, seed):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "linear":
        from sklearn.linear_model import RidgeCV
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    if kind == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=-1)
    if kind == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, random_state=seed, n_jobs=1)
    if kind == "mlp":
        from sklearn.neural_network import MLPRegressor
        return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-2,
                                                            max_iter=3000, early_stopping=True,
                                                            validation_fraction=0.15, n_iter_no_change=50,
                                                            random_state=seed))
    raise ValueError(kind)


def fit_predict_cnn(Xtr, ytr, Xte, seed, epochs):
    """Small 2D CNN on (frames x bins x antennas). / 小さな 2 次元 CNN"""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    tf.keras.utils.set_random_seed(seed)
    mu = Xtr.mean(axis=(0, 1, 2), keepdims=True)
    sd = Xtr.std(axis=(0, 1, 2), keepdims=True) + 1e-6
    ym, ys = ytr.mean(), ytr.std() + 1e-9
    L = tf.keras.layers
    model = tf.keras.Sequential([
        L.Input(shape=Xtr.shape[1:]),
        L.Conv2D(16, 3, padding="same"), L.BatchNormalization(), L.ReLU(), L.MaxPooling2D(2),
        L.Conv2D(32, 3, padding="same"), L.BatchNormalization(), L.ReLU(), L.MaxPooling2D(2),
        L.Conv2D(64, 3, padding="same"), L.BatchNormalization(), L.ReLU(),
        L.GlobalAveragePooling2D(), L.Dropout(0.3), L.Dense(32, activation="relu"), L.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    cb = [tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)]
    model.fit((Xtr - mu) / sd, (ytr - ym) / ys, validation_split=0.15, epochs=epochs, batch_size=16,
              verbose=0, callbacks=cb, shuffle=True)
    return model.predict((Xte - mu) / sd, verbose=0).ravel() * ys + ym


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    err = p - y
    ss = np.sum((y - y.mean()) ** 2)
    r = float(np.corrcoef(y, p)[0, 1]) if np.std(p) > 0 and np.std(y) > 0 else float("nan")
    return dict(MAE=float(np.mean(np.abs(err))), RMSE=float(np.sqrt(np.mean(err ** 2))),
                R2=float(1 - np.sum(err ** 2) / ss) if ss > 0 else float("nan"), r=r)


# ============================================================ run
def run(opts):
    out = []
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = opts.out or os.path.join(HERE, "outputs", f"regression_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    log(f"=== ドップラー → ENMO の回帰 ({stamp}) ===", out)
    trial_dirs = sorted(d for d in glob.glob(os.path.join(opts.input, "*")) if os.path.isdir(d))
    if not trial_dirs:
        raise RegError(f"{opts.input} に試行のフォルダがありません (README.md の置き方を参照)")
    variants = list(VARIANT_DIRS) if opts.doppler == "all" else [opts.doppler]
    models = [m.strip() for m in opts.models.split(",") if m.strip()]
    bad = [m for m in models if m not in ALL_MODELS]
    if bad:
        raise RegError(f"不明なモデル: {', '.join(bad)}（使えるのは {', '.join(ALL_MODELS)}）")
    feats = [f.strip().upper() for f in opts.features.split(",") if f.strip()]
    results, pred_rows = [], []
    windows = opts.windows
    for variant, window in [(v, w) for v in variants for w in windows]:
        log(f"\n[ドップラー: {variant}（{VARIANT_DIRS[variant]}/）、区間 {window:g} 秒]", out)
        try:
            trials = [build_trial(d, variant, window, out) for d in trial_dirs]
        except RegError as e:
            if opts.doppler == "all":
                log(f"  この版は飛ばします: {e}", out)
                continue
            raise
        # folds / 学習とテストの分け方
        if len(trials) >= 2:
            groups = [np.full(len(tr["y"]), k) for k, tr in enumerate(trials)]
            how = f"試行ごと（{len(trials)} 試行。1 試行をテスト、残りで学習）"
        else:
            n = len(trials[0]["y"])
            groups = [np.minimum((np.arange(n) * opts.blocks) // n, opts.blocks - 1)]
            how = f"試行が 1 本なので、時間を {opts.blocks} つのかたまりに分けて交互にテスト（参考値）"
        log(f"  評価: {how}", out)
        g = np.concatenate(groups)
        XA = np.concatenate([tr["XA"] for tr in trials])
        XB = np.concatenate([tr["XB"] for tr in trials])
        XC = np.concatenate([tr["XC"] for tr in trials])
        y = np.concatenate([tr["y"] for tr in trials])
        tname = np.concatenate([[tr["name"]] * len(tr["y"]) for tr in trials])
        win = np.concatenate([tr["win"] for tr in trials])
        _dump_features(os.path.join(out_dir, f"features_{variant}_{window:g}s.csv"), tname, win, XA, XB, y,
                       trials[0]["v"])
        configs = [("baseline", "-")]
        for m in models:
            if m == "cnn":
                configs.append(("cnn", "Doppler 2D"))
            else:
                configs += [(m, f) for f in feats]
        for model, feat in configs:
            t_start = time.time()
            pred = np.full(len(y), np.nan)
            for k in np.unique(g):
                tr_i, te_i = g != k, g == k
                if model == "baseline":
                    pred[te_i] = y[tr_i].mean()
                elif model == "cnn":
                    try:
                        pred[te_i] = fit_predict_cnn(XC[tr_i], y[tr_i], XC[te_i], opts.seed, opts.epochs)
                    except ImportError:
                        log("  CNN: TensorFlow がないので飛ばします (pip install tensorflow)", out)
                        pred[:] = np.nan
                        break
                else:
                    X = XA if feat == "A" else XB
                    est = make_tabular(model, opts.seed)
                    est.fit(X[tr_i], y[tr_i])
                    pred[te_i] = est.predict(X[te_i])
            if np.isnan(pred).all():
                continue
            mt = metrics(y, pred)
            per = {tn: metrics(y[tname == tn], pred[tname == tn]) for tn in dict.fromkeys(tname)}
            label = "平均値で予測（基準）" if model == "baseline" else MODEL_NAMES[model]
            results.append(dict(variant=variant, window=window, model=model, label=label, features=feat, **mt,
                                per_trial=per,
                                seconds=time.time() - t_start, pred=pred, y=y, tname=tname, win=win))
            log(f"  {label:16s} 特徴量 {feat:10s}: MAE {mt['MAE']:7.2f} mg  RMSE {mt['RMSE']:7.2f}  "
                f"R² {mt['R2']:6.3f}  r {mt['r']:6.3f}  ({time.time() - t_start:.0f} s)", out)
            for i in range(len(y)):
                pred_rows.append([variant, f"{window:g}", model, feat, tname[i], f"{win[i][0]:.3f}", f"{win[i][1]:.3f}",
                                  f"{y[i]:.3f}", f"{pred[i]:.3f}"])
    if not results:
        raise RegError("どの版でも計算できませんでした")
    _save(out_dir, results, pred_rows, opts, out)
    if not opts.no_plots:
        _plots(out_dir, results)
    log(f"\n保存しました: {out_dir}", out)
    with open(os.path.join(out_dir, "log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    return results, out_dir


def _dump_features(path, tname, win, XA, XB, y, v):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        na = XA.shape[1] // 2
        w.writerow(["trial", "start_csi(s)", "end_csi(s)", TARGET] + B_NAMES_full()
                   + [f"A_ant{1 + j // na}_v{v[j % na]:+.3f}" for j in range(XA.shape[1])])
        for i in range(len(y)):
            w.writerow([tname[i], f"{win[i][0]:.3f}", f"{win[i][1]:.3f}", f"{y[i]:.3f}"]
                       + [f"{x:.6g}" for x in XB[i]] + [f"{x:.6g}" for x in XA[i]])


def B_NAMES_full():
    return [f"B_ant{a}_{n}" for a in (1, 2) for n in B_NAMES]


def _save(out_dir, results, pred_rows, opts, out):
    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doppler", "window(s)", "model", "features", "MAE(mg)", "RMSE(mg)", "R2", "r", "seconds"])
        for r in results:
            w.writerow([r["variant"], f"{r['window']:g}", r["label"], r["features"], f"{r['MAE']:.3f}", f"{r['RMSE']:.3f}",
                        f"{r['R2']:.4f}", f"{r['r']:.4f}", f"{r['seconds']:.1f}"])
    with open(os.path.join(out_dir, "metrics_by_trial.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doppler", "window(s)", "model", "features", "trial", "MAE(mg)", "RMSE(mg)", "R2", "r"])
        for r in results:
            for tn, m in r["per_trial"].items():
                w.writerow([r["variant"], f"{r['window']:g}", r["label"], r["features"], tn, f"{m['MAE']:.3f}", f"{m['RMSE']:.3f}",
                            f"{m['R2']:.4f}", f"{m['r']:.4f}"])
    with open(os.path.join(out_dir, "predictions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doppler", "window(s)", "model", "features", "trial", "start_csi(s)", "end_csi(s)",
                    "true_ENMO(mg)", "pred_ENMO(mg)"])
        w.writerows(pred_rows)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(opts), f, ensure_ascii=False, indent=2)
    for wl in dict.fromkeys(r["window"] for r in results):
        best = min((r for r in results if r["model"] != "baseline" and r["window"] == wl), key=lambda r: r["MAE"],
                   default=None)
        if best:
            log(f"\n{wl:g} 秒の区間で一番誤差が小さかったのは {best['label']}（特徴量 {best['features']}、"
                f"ドップラー {best['variant']}）: MAE {best['MAE']:.2f} mg、R² {best['R2']:.3f}", out)


# ============================================================ plots
def _setup_plt():
    import warnings
    import matplotlib
    matplotlib.use("Agg")
    warnings.filterwarnings("ignore", message="Glyph .* missing from font")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    jp = [f for f in ("Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP", "IPAexGothic", "Hiragino Sans",
                      "WenQuanYi Zen Hei") if f in have]
    plt.rcParams["font.family"] = jp + ["DejaVu Sans"]
    return plt


def _plots(out_dir, results):
    plt = _setup_plt()
    colors = {"baseline": "#999999", "linear": "#2A6FDB", "rf": "#3A9D5D", "xgboost": "#D9822B",
              "mlp": "#B5487A", "cnn": "#6A4FB3"}
    # one figure per model: each trial's true vs predicted over time + scatter
    # モデルごと: 試行ごとの正解と予測の時間変化 + 散布図
    for r in results:
        if r["model"] == "baseline":
            continue
        names = list(dict.fromkeys(r["tname"]))
        fig = plt.figure(figsize=(14, 2.3 * len(names) + 1.5))
        gs = fig.add_gridspec(len(names), 3)
        for i, tn in enumerate(names):
            m = r["tname"] == tn
            mid = r["win"][m].mean(axis=1)
            ax = fig.add_subplot(gs[i, :2])
            ax.plot(mid, r["y"][m], color="#333333", lw=1.4, label="正解（ENMO）")
            ax.plot(mid, r["pred"][m], color=colors[r["model"]], lw=1.2, label="予測")
            pt = r["per_trial"][tn]
            ax.set_title(f"{tn}  MAE {pt['MAE']:.1f} mg / R² {pt['R2']:.2f}", loc="left", fontsize=10)
            ax.set_ylabel("ENMO [mg]")
            if i == 0:
                ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=9, frameon=False)
            if i == len(names) - 1:
                ax.set_xlabel("Time [s]")
        ax = fig.add_subplot(gs[:, 2])
        lo, hi = min(r["y"].min(), r["pred"].min()), max(r["y"].max(), r["pred"].max())
        for tn in names:
            m = r["tname"] == tn
            ax.scatter(r["y"][m], r["pred"][m], s=12, alpha=0.7, label=tn)
        ax.plot([lo, hi], [lo, hi], color="#333333", lw=0.8, ls="--")
        ax.set_xlabel("正解 [mg]")
        ax.set_ylabel("予測 [mg]")
        ax.set_title(f"全体: MAE {r['MAE']:.1f} mg / R² {r['R2']:.2f} / r {r['r']:.2f}", fontsize=10)
        ax.legend(fontsize=8)
        fig.suptitle(f"{r['label']}（特徴量 {r['features']}、ドップラー {r['variant']}、区間 {r['window']:g} 秒）")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"pred_{r['variant']}_{r['window']:g}s_{r['model']}_"
                                          f"{r['features'].split()[0]}.png"), dpi=110)
        plt.close(fig)
    # comparison / 比較
    groups = list(dict.fromkeys((r["variant"], r["window"]) for r in results))
    fig, ax = plt.subplots(len(groups), 2, figsize=(14, 4.2 * len(groups)), squeeze=False)
    for vi, (var, wl) in enumerate(groups):
        rs = [r for r in results if r["variant"] == var and r["window"] == wl]
        short = {"baseline": "基準\n(平均値)", "linear": "線形", "rf": "RF", "xgboost": "XGBoost",
                 "mlp": "MLP", "cnn": "CNN"}
        labels = [short[r["model"]] + ("" if r["model"] in ("baseline", "cnn") else f"\n特徴量{r['features']}")
                  for r in rs]
        x = np.arange(len(rs))
        cols = [colors[r["model"]] for r in rs]
        for j, (key, unit) in enumerate((("MAE", " [mg]（小さいほど良い）"), ("R2", "（1 に近いほど良い）"))):
            vals = [r[key] for r in rs]
            ax[vi, j].bar(x, vals, color=cols)
            for xi, val in zip(x, vals):
                ax[vi, j].text(xi, val, f"{val:.2f}" if key == "R2" else f"{val:.1f}", ha="center",
                               va="bottom" if val >= 0 else "top", fontsize=8)
            ax[vi, j].set_xticks(x)
            ax[vi, j].set_xticklabels(labels, fontsize=9)
            ax[vi, j].set_ylabel(("R²" if key == "R2" else key) + unit)
            ax[vi, j].set_title(f"ドップラー: {var}、区間 {wl:g} 秒", loc="left", fontsize=10)
            if key == "R2":
                ax[vi, j].axhline(0, color="#333333", lw=0.8)
    fig.suptitle("モデルの比較（試行ごとに分けた評価。RF = ランダムフォレスト、線形 = 線形回帰 (Ridge)）")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "comparison.png"), dpi=110)
    plt.close(fig)


# ============================================================ CLI
def parse_windows(text):
    try:
        ws = [float(x) for x in str(text).split(",") if x.strip()]
    except ValueError:
        raise RegError(f"--windows は数字をカンマで区切って指定します（{text}）")
    if not ws or any(w <= 0 for w in ws):
        raise RegError(f"--windows の値が正しくありません（{text}）")
    return list(dict.fromkeys(ws))


def main(argv=None):
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=os.path.join(HERE, "regression_input"), help="入力フォルダ（既定 regression_input）")
    ap.add_argument("--out", default=None, help="保存先（既定 outputs/regression_<日時>）")
    ap.add_argument("--doppler", choices=list(VARIANT_DIRS) + ["all"], default="nowinnorm",
                    help="使うドップラーの版（既定 nowinnorm。all で 3 つとも）")
    ap.add_argument("--models", default=",".join(ALL_MODELS), help=f"使うモデル（既定 {','.join(ALL_MODELS)}）")
    ap.add_argument("--features", default="A,B", help="CNN 以外のモデルの特徴量（A, B。既定 A,B）")
    ap.add_argument("--windows", default="1,5", help="区間の長さ [s]（カンマ区切り。既定 1,5）")
    ap.add_argument("--blocks", type=int, default=5, help="試行が 1 本のとき、時間を分けるかたまりの数（既定 5）")
    ap.add_argument("--epochs", type=int, default=200, help="CNN の最大エポック数（既定 200。早期終了あり）")
    ap.add_argument("--seed", type=int, default=0, help="乱数の種（既定 0）")
    ap.add_argument("--no_plots", action="store_true", help="図を作らない")
    opts = ap.parse_args(argv)
    try:
        opts.windows = parse_windows(opts.windows)
        run(opts)
    except RegError as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
