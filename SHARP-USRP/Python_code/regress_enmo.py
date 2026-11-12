"""
    ドップラーから腰の加速度の指標 (ENMO) を推定する回帰モデルを作り、比べる。

    正解: 加速度計算ソフト (witmotion_wt011 の compute_power.py) が出す ENMO
          (max(合成加速度 - 1 g, 0) [mG]。mG = ミリ G = 0.001 g。ミリグラムではない) の 1 秒ごとの平均。--target で選ぶ:
            waist (既定) = 腰の ENMO (列 ENMO(mG))
            body         = 全身の ENMO (腰・大腿・下腿・手首の 4 台の ENMO の合計、列 ENMO_body(mG)。
                           4 台そろった試行だけ。列がない試行は使わない)
    入力: SHARP-USRP のドップラー (doppler_traces*/ の数値の配列。PNG ではない)

    モデル (5 種類):
      線形回帰 (Ridge)    : 特徴量 A / B（C や A+C などは --features で指定したときだけ）
      ランダムフォレスト  : 同上
      XGBoost             : 同上
      MLP                 : 同上
      SVR                 : 同上（RBF カーネル）
      ガウス過程回帰       : 同上（推定の幅も出す）
      CNN                 : 1 秒分のドップラーの 2 次元配列 (時間 x 速度 x アンテナ 2 本)
    特徴量 F (既定) = 動いている速度ビン (|v| >= 0.65 m/s、--static_v) で足したドップラーを 50 Hz にそろえ、
               Savitzky-Golay (2 次式、微分の窓 0.2 秒、--deriv_window) で時間微分し、1 秒ごとに |微分| を平均
               (アンテナごと)。F1: log10 Σ_動 D、F2a: Σ_動 |v| D ÷ Σ_静止 D、F2b: Σ_動 |v| D ÷ Σ_動 D (平均の速さ)
               あわせて、ENMO の小さい区間の平均スペクトルで静止成分の広がりを確かめる (static_spectrum_*.png)
    以下の特徴量 A〜E は前に作ったもの (--features で指定すれば使える)
    特徴量 A = 1 秒の中で速度ビンごとに平均したスペクトル (log10, アンテナ 2 本で 200 個)
    特徴量 B = 物理に基づく特徴量 (速さの平均・2 乗平均・広がり・最大速度・足踏みの周期など,
               アンテナごと)
    特徴量 C = 加速度に当たる特徴量 (ENMO は加速度なので、速さの時間微分を入れる。アンテナごと):
               フレームごとの平均速度 (向きつき) と 95 % の速さを 6 Hz で平滑化して微分した
               |dv/dt| の平均・2 乗平均の平方根・95 % 値、スペクトルのフレーム間の変化の速さ
    どのモデルも出力は区間ごとの ENMO の値 (数値)。最後に予測と正解を図にする。

    評価: 試行ごとに分ける (1 試行をテスト、残りで学習、を全試行について繰り返す)。
          隣り合う 1 秒どうしはよく似ているので、ランダムに分けると精度が高く出すぎるため。
          試行が 1 本しかないときは、時間を 5 つのかたまりに分けて同じことをする。

    入力フォルダ (regression_input/ の README.md を参照。次の 2 通りのどちらでもよい):
      regression_input/A1/imu/<試行名>/(imu/)<試行名>_power_timeseries.csv と
      regression_input/A1/doppler_traces_nowinnorm/<全試行のドップラー>  (名前か batch_pairs_*.csv で対応づける)
    または
      regression_input/<試行名>/
          <名前>_power_timeseries.csv  (正解データ。加速度計算ソフトの出力。imu/ の中に置いてもよい)
          doppler_traces/            *antenna1*.txt, *antenna2*.txt (+ _time.npy, _meta.json)
          doppler_traces_nowinnorm/  同上 (--save_unnormalized の版)
          doppler_traces_raw/        同上

    例:
      python regress_enmo.py                        # nowinnorm 版で 5 モデル、1 秒ごと
      python regress_enmo.py --doppler all          # 3 つの版それぞれで比べる
      python regress_enmo.py --windows 1            # 1 秒の区間だけ
      python regress_enmo.py --models linear,rf     # 一部のモデルだけ
      python regress_enmo.py --target body          # 全身の ENMO を正解にする

    Copyright (C) 2026 Heisuke Takeda
    Released under the GNU GPL v3 (see LICENSE).
"""

import argparse
import csv
import glob
import json
import os
import pickle
import re
import sys
import time
import warnings
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VARIANT_DIRS = {"normalized": "doppler_traces", "nowinnorm": "doppler_traces_nowinnorm",
                "raw": "doppler_traces_raw"}
MODEL_NAMES = {"linear": "線形回帰 (Ridge)", "rf": "ランダムフォレスト", "xgboost": "XGBoost",
               "mlp": "MLP", "svr": "SVR", "gpr": "ガウス過程回帰", "cnn": "CNN",
               "cnn_shuffle": "CNN（時間の順番を入れ替え）", "cnn_timemean": "CNN（時間で平均）",
               "cnn1d": "1 次元 CNN（時間方向）", "cnn3s": "CNN（前後 1 秒も入力）", "cnn5s": "CNN（前後 2 秒も入力）"}
ALL_MODELS = ["linear", "rf", "xgboost", "mlp", "svr", "gpr", "cnn"]
# extra CNNs for checking why the CNN works (--cnn_study) / CNN が効く理由を確かめるための CNN
STUDY_MODELS = ["cnn_shuffle", "cnn_timemean", "cnn1d", "cnn3s"]
EXTRA_MODELS = ["cnn5s"]   # only when named in --models (heavy) / --models で指定したときだけ（重い）
CNN_KINDS = ["cnn"] + STUDY_MODELS + EXTRA_MODELS
CONTEXT_S = {"cnn3s": 1.0, "cnn5s": 2.0}   # seconds of Doppler before and after the window / 区間の前後に足す秒数
CNN_FEAT = {"cnn": "Doppler 2D", "cnn_shuffle": "Doppler 2D 時間入替", "cnn_timemean": "Doppler 2D 時間平均",
            "cnn1d": "Doppler 1D", "cnn3s": "Doppler 2D 3秒", "cnn5s": "Doppler 2D 5秒"}
CNN_FRAMES = 64          # CNN input frames per second of window (at most 320) / CNN に入れる 1 秒あたりの時間フレーム数（最大 320）
TARGET = "ENMO(mG)"      # target column in the calculator's output (waist) / 正解の列（腰）
# --target: which ENMO of the acceleration calculator is the ground truth / どの ENMO を正解にするか
TARGETS = {"waist": "ENMO(mG)", "body": "ENMO_body(mG)"}
TARGET_JA = {"waist": "腰の ENMO", "body": "全身の ENMO（腰・大腿・下腿・手首の合計）"}
ACC_RATE = 50.0          # feature C: Doppler traces are averaged onto this grid [Hz] / 特徴量 C の時間格子
ACC_FC = 6.0             # feature C: low-pass before differentiating [Hz] / 特徴量 C: 微分の前の平滑化
EPS = 1e-12
# MAPE: windows whose true ENMO is below this are left out (ENMO is ~0 when still, so error / truth blows up)
# MAPE（平均絶対パーセント誤差）: 正解がこれ未満の区間は除く（静止で ENMO ≈ 0 になり、誤差 ÷ 正解が発散するため）
MAPE_MIN = 50.0


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


def load_doppler(txt, out):
    """One antenna's Doppler: (frames x bins) array, frame times on the CSI axis [s], velocity axis [m/s].
    1 本のアンテナのドップラー: (時間フレーム x 速度ビン) の配列、各フレームの CSI の時刻、速度軸"""
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


def _target_file(spec, window):
    if spec.get("ts"):
        return spec["ts"], "timeseries"
    if spec.get("win", {}).get(window):
        return spec["win"][window], "windows"
    return None, None


def _column(header, column):
    """The ENMO column as written in the file: the calculator before v2.1 wrote the unit as mg instead of mG.
    ファイルの中の ENMO の列名（加速度計算ソフト v2.1 より前は単位を mG ではなく mg と書いていた）"""
    if column in header:
        return column
    old = column.replace("(mG)", "(mg)")
    return old if old in header else None


def has_target(spec, window, column):
    """True if the trial's ground-truth file has this ENMO column. / 正解のファイルにこの ENMO の列があるか"""
    path, _ = _target_file(spec, window)
    if not path:
        return True   # reported later by load_target / 後で load_target が知らせる
    with open(path, encoding="utf-8") as f:
        return _column(next(csv.reader(f), []), column) is not None


def load_target(spec, window, column=TARGET):
    """ENMO per sample (t_csi, value), or per window. / ENMO（各時刻、または区間ごと）"""
    if spec.get("ts"):
        path, kind = spec["ts"], "timeseries"
    elif spec.get("win", {}).get(window):
        path, kind = spec["win"][window], "windows"
    else:
        raise RegError(f"{spec['name']}: *_power_timeseries.csv (または *_windows_{window:g}s.csv) がありません")
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    found = _column(list(rows[0]), column) if rows else None
    if rows and column != TARGET and found is None:
        raise RegError(f"{os.path.basename(path)} に全身の ENMO の列 ({column}) がありません。センサが 4 台そろっていない"
                       "試行か、加速度計算ソフトが古いです（v2.0 で計算し直してください）")
    if not rows or found is None:
        raise RegError(f"{os.path.basename(path)} に ENMO の列 ({column}) がありません。"
                       "加速度計算ソフトを最新にして計算し直してください")
    if kind == "timeseries":
        t = np.array([float(r["t_csi(s)"]) for r in rows])
        y = np.array([float(r[found]) for r in rows])
        return dict(kind=kind, t=t, y=y, file=os.path.basename(path))
    a = np.array([float(r["start_csi(s)"]) for r in rows])
    b = np.array([float(r["end_csi(s)"]) for r in rows])
    y = np.array([float(r[found]) for r in rows])
    return dict(kind=kind, a=a, b=b, y=y, file=os.path.basename(path))


# ============================================================ finding the trials
TS_SUFFIX = "_power_timeseries.csv"


def _antenna(name):
    m = re.search(r"antenna([12])", name, flags=re.I)
    return f"antenna{m.group(1)}" if m else None


def _dop_key(name):
    """'A1_XR1_antenna1_WAX202_aligned_CSI_stream_0' -> 'A1_XR1_antenna#_WAX202_aligned_CSI_stream_0'"""
    return re.sub(r"antenna[12]", "antenna#", name, flags=re.I)


def _dop_prefix(key):
    """Part of the Doppler name before the antenna (usually the trial / CSI name). / アンテナより前の部分"""
    return re.split(r"_?antenna#", key, flags=re.I)[0].strip("_")


EXCLUDE_DEFAULT = "その他"   # folders under the input folder that are not read / 読まないフォルダ


def discover(root, out, pairs_csv=None, exclude=EXCLUDE_DEFAULT):
    """Find every trial: its ground truth and its Doppler files, in either layout.
    試行ごとに、正解データとドップラーを探す（どちらの置き方でもよい）

      (a) one folder per trial / 試行ごとのフォルダ:
            <root>/<trial>/<name>_power_timeseries.csv (or <trial>/imu/...), <root>/<trial>/doppler_traces_nowinnorm/*.txt
      (b) a dataset folder with everything mixed / データセットのフォルダにまとめて:
            <root>/A1/imu/<trial>/(imu/)<name>_power_timeseries.csv, <root>/A1/doppler_traces_nowinnorm/*.txt (all trials)
    Ground truth and Doppler are matched by folder (a), by the pairing list of the batch run (batch_pairs_*.csv),
    or by name. / 正解とドップラーは、同じフォルダ (a)、一括処理の組み合わせの一覧、名前の順で対応づける
    """
    skip = {x.strip().lower() for x in str(exclude or "").split(",") if x.strip()}

    def excluded(path):  # inside a folder that is not read / 読まないフォルダの中か
        rel = os.path.relpath(path, root)
        return any(part.lower() in skip for part in rel.split(os.sep)[:-1])

    if skip:
        log(f"  読まないフォルダ: {', '.join(sorted(skip))}（--exclude で変更）", out)
    truths = {}
    for p in sorted(glob.glob(os.path.join(root, "**", "*" + TS_SUFFIX), recursive=True)):
        if excluded(p):
            continue
        name = os.path.basename(p)[:-len(TS_SUFFIX)]
        if name in truths and truths[name].get("ts"):
            raise RegError(f"正解データ {name}{TS_SUFFIX} が 2 か所にあります: {truths[name]['ts']} と {p}")
        truths.setdefault(name, dict(name=name, win={}))["ts"] = p
    for p in sorted(glob.glob(os.path.join(root, "**", "*_windows_*s.csv"), recursive=True)):
        if excluded(p):
            continue
        m = re.match(r"(.+)_windows_([\d.]+)s\.csv$", os.path.basename(p))
        if m:
            truths.setdefault(m.group(1), dict(name=m.group(1), win={}))["win"][float(m.group(2))] = p
    def dataset_of(path):  # first folder under the input folder (A1, A2, ... or the trial folder) / データセット名
        rel = os.path.relpath(path, root)
        return "" if rel in (".", "") else rel.split(os.sep)[0]

    for t in truths.values():
        d = os.path.dirname(t.get("ts") or next(iter(t["win"].values())))
        t["dir"] = os.path.dirname(d) if os.path.basename(d).lower() == "imu" else d
        t["ds"] = dataset_of(t["dir"])

    # Doppler groups (antenna1 + antenna2 of one capture), keyed by the name without the antenna number
    # ドップラーのまとまり（同じ収集の antenna1 と antenna2）。アンテナ番号を除いた名前で分ける
    groups = {}
    for variant, folder in VARIANT_DIRS.items():
        for d in sorted(glob.glob(os.path.join(root, "**", folder), recursive=True)):
            if not os.path.isdir(d) or os.path.basename(d) != folder or excluded(os.path.join(d, "x")):
                continue
            for txt in sorted(glob.glob(os.path.join(d, "*.txt"))):
                stem = os.path.basename(txt)[:-4]
                ant = _antenna(stem)
                if not ant:
                    continue
                key = (os.path.dirname(d), _dop_key(stem))
                g = groups.setdefault(key, dict(key=key[1], parent=key[0], ds=dataset_of(key[0]), files={}))
                g["files"].setdefault(variant, {})[ant] = txt

    # pairing list of the batch run: IMU CSV name -> CSI name / 一括処理の組み合わせ（IMU の名前 → CSI の名前）
    mapping = {}
    # every pairing list found (A1, A2, ... may each have one; newer files win) / 見つかった一覧をすべて使う
    lists = [pairs_csv] if pairs_csv else sorted(
        (p for p in glob.glob(os.path.join(root, "**", "batch_pairs_*.csv"), recursive=True) if not excluded(p)),
        key=os.path.getmtime)
    for pc in lists:
        n0 = len(mapping)
        with open(pc, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if r.get("imu") and r.get("csi"):
                    mapping[os.path.splitext(r["imu"].strip())[0]] = os.path.splitext(r["csi"].strip())[0]
        log(f"  組み合わせの一覧を使います: {pc}（{len(mapping) - n0} 組）", out)
    ds_with_doppler = {g["ds"] for g in groups.values()}

    cand = []
    for tn, t in truths.items():
        for gk, g in groups.items():
            pre = _dop_prefix(g["key"])
            if os.path.normcase(g["parent"]) == os.path.normcase(t["dir"]):
                cand.append((3, 1.0, tn, gk, "同じフォルダ"))
            elif tn in mapping and _dop_key(mapping[tn]).lower() in g["key"].lower():
                cand.append((2, 1.0, tn, gk, "組み合わせの一覧"))
            elif t["ds"] != g["ds"] and t["ds"] in ds_with_doppler:
                continue  # never pair by name across datasets (A1_XR1 vs A2_XR1) / 名前ではデータセットをまたがない
            else:
                # by name only when one name is the other plus a "_..." suffix: a similar name such as
                # A3_XL1 vs A3_XR1 is another trial, never a match / 名前は「片方 + _…」のときだけ。
                # A3_XL1 と A3_XR1 のような似た名前は別の試行なので組にしない
                a, b = tn.lower(), pre.lower()
                if b and (a == b or a.startswith(b + "_") or b.startswith(a + "_")):
                    cand.append((1, 1.0 if a == b else 0.9, tn, gk, "名前"))
    cand.sort(key=lambda c: (c[0], c[1]), reverse=True)
    used_t, used_g, specs = set(), set(), []
    for _, _, tn, gk, how in cand:
        if tn in used_t or gk in used_g:
            continue
        used_t.add(tn)
        used_g.add(gk)
        specs.append(dict(truths[tn], doppler=groups[gk]["files"], dop_name=groups[gk]["key"], how=how))
    specs.sort(key=lambda s_: s_["name"])
    log(f"  試行: {len(specs)} 本（正解データとドップラーの組）", out)
    for sp in specs:
        log(f"    {sp['name']}  ↔  {sp['dop_name']}  （{sp['how']}で対応" +
            (f"、データセット {sp['ds']}）" if sp["ds"] and sp["ds"] != sp["name"] else "）"), out)
    for tn in sorted(set(truths) - used_t):
        log(f"  注意: ドップラーが見つからない正解データ: {tn}（飛ばします）", out)
    for gk in sorted(set(groups) - used_g):
        log(f"  注意: 正解データが見つからないドップラー: {groups[gk]['key']}（飛ばします）", out)
    # datasets that have ground truth but no Doppler at all: say what is there / 正解だけでドップラーがないデータセット
    for ds in sorted({truths[tn]["ds"] for tn in set(truths) - used_t} - ds_with_doppler - {""}):
        base = os.path.join(root, ds)
        dirs = sorted({os.path.relpath(d, base) for d in glob.glob(os.path.join(base, "**", "*"), recursive=True)
                       if os.path.isdir(d) and "doppler" in os.path.basename(d).lower()})
        txts = sorted(glob.glob(os.path.join(base, "**", "*.txt"), recursive=True))
        no_ant = [os.path.basename(t_) for t_ in txts if not _antenna(os.path.basename(t_))]
        log(f"  注意: {ds} にドップラーが見つかりません。ドップラーは {ds}/{VARIANT_DIRS['nowinnorm']}/ などのフォルダに、"
            f"名前に antenna1 / antenna2 が入った .txt（_time.npy・_meta.json も）で置いてください", out)
        log(f"        {ds} の中の doppler の付くフォルダ: {', '.join(dirs) or 'なし'}／.txt: {len(txts)} 個"
            + (f"（antenna の入っていない名前: {', '.join(no_ant[:4])}{' …' if len(no_ant) > 4 else ''}）" if no_ant else ""), out)
    if not specs:
        raise RegError(f"{root} に、正解データとドップラーの組が見つかりません (README.md の置き方を参照)")
    return specs


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


def accel_traces(d):
    """Acceleration-like traces of one antenna for feature set C (computed once per trial).
    The Doppler frames (one per packet) are averaged onto a 50 Hz grid, low-passed at 6 Hz (differentiating
    amplifies noise) and differentiated.
    特徴量 C のための、加速度に当たる時系列（アンテナ 1 本・試行ごとに 1 回）。ドップラーのフレーム（パケットごと）を
    50 Hz の格子に平均し、6 Hz で平滑化してから（微分は雑音を大きくするため）微分する
    Returns dict(tu, a_mean, a_v95, flux): grid times, d/dt of the signed mean velocity, d/dt of the 95 % speed
    [m/s^2], and how fast the speed distribution changes [1/s]."""
    from scipy.signal import butter, filtfilt
    S, t, v = d["S"], d["t"], d["v"]
    P = np.clip(S, 0, None)
    av = np.abs(v)
    move = av > 1.5 * np.min(av[av > 0]) if np.any(av > 0) else np.ones_like(av, bool)
    w = P[:, move] / (P[:, move].sum(axis=1, keepdims=True) + EPS)   # speed distribution / 速さの分布
    vs = (w * v[move]).sum(axis=1)                                  # signed mean velocity / 向きつき平均速度
    order = np.argsort(av[move])
    cs = np.cumsum(w[:, order], axis=1)
    v_sorted = av[move][order]
    v95 = v_sorted[np.minimum(np.argmax(cs >= 0.95, axis=1), len(v_sorted) - 1)]
    dt = 1.0 / ACC_RATE
    tu = np.arange(t[0], t[-1] + 1e-9, dt)
    idx = np.clip(np.round((t - tu[0]) / dt).astype(int), 0, len(tu) - 1)
    cnt = np.bincount(idx, minlength=len(tu)).astype(float)
    have = cnt > 0

    def to_grid(x):  # average of the frames in each cell; empty cells interpolated / 格子ごとの平均（空はつなぐ）
        x = np.asarray(x, float)
        if x.ndim == 1:
            g = np.bincount(idx, weights=x, minlength=len(tu))[have] / cnt[have]
            return np.interp(tu, tu[have], g)
        out = np.empty((len(tu), x.shape[1]))
        for j in range(x.shape[1]):
            g = np.bincount(idx, weights=x[:, j], minlength=len(tu))[have] / cnt[have]
            out[:, j] = np.interp(tu, tu[have], g)
        return out

    vs_g, v95_g, w_g = to_grid(vs), to_grid(v95), to_grid(w)
    if len(tu) > 15:
        b, a = butter(2, ACC_FC / (ACC_RATE / 2))
        vs_g, v95_g = filtfilt(b, a, vs_g), filtfilt(b, a, v95_g)
        w_g = filtfilt(b, a, w_g, axis=0)
    a_mean = np.gradient(vs_g, dt)
    a_v95 = np.gradient(v95_g, dt)
    flux = np.concatenate([[0.0], np.abs(np.diff(w_g, axis=0)).sum(axis=1) / dt])
    return dict(tu=tu, a_mean=a_mean, a_v95=a_v95, flux=flux)


def accel_features(tr, a, b):
    """Feature set C for one antenna and one window. / 特徴量 C（アンテナ 1 本・1 区間）"""
    m = (tr["tu"] >= a) & (tr["tu"] < b)
    if m.sum() < 2:
        return np.full(len(C_NAMES), np.nan)
    am, a95, fl = np.abs(tr["a_mean"][m]), np.abs(tr["a_v95"][m]), tr["flux"][m]
    rms = lambda x: float(np.sqrt(np.mean(x ** 2)))  # noqa: E731
    return np.array([np.mean(am), rms(am), np.percentile(am, 95),
                     np.mean(a95), rms(a95), np.percentile(a95, 95),
                     np.mean(fl)])


C_NAMES = ["acc_mean_abs", "acc_mean_rms", "acc_mean_p95", "acc_v95_abs", "acc_v95_rms", "acc_v95_p95",
           "spec_change_rate"]


# ---- feature F: Doppler summed over the moving velocity bins -> time derivative -> 1 s mean of |d/dt| ----
# 特徴量 F: 動いている速度ビンで足したドップラー → 時間微分 → 1 秒ごとに |微分| の平均
F_PARTS = ("F1", "F2A", "F2B")
F_DESC = {"F1": "log10 Σ_動 D の変化の速さ [1/s]", "F2A": "Σ_動 |v| D ÷ Σ_静止 D の変化の速さ（運動量 → 力に近い）",
          "F2B": "平均の速さ Σ_動 |v| D ÷ Σ_動 D の変化の速さ [m/s²]（加速度相当）"}
F_RATE = 50.0              # the series are averaged onto this grid before differentiating [Hz] / 微分の前にそろえる格子
STATIC_V = 0.65            # |v| below this is the static path (main lobe of the Hann window, about 2 x resolution) [m/s]
DERIV_WINDOW = 0.2         # Savitzky-Golay derivative window [s] (about the Doppler's own time resolution, 31 packets)


def deriv_traces(d, static_v, deriv_windows):
    """Feature F series of one antenna, over the whole trial (computed once per trial).
    Per frame: P_mov = Σ_{|v|>=static_v} D, P_static = Σ_{|v|<static_v} D, S1 = Σ_{|v|>=static_v} |v| D;
      F1 = log10 P_mov, F2A = S1 / P_static, F2B = S1 / P_mov.
    Frames (one per packet, uneven) are averaged onto a 50 Hz grid, then differentiated with a Savitzky-Golay filter
    (2nd order) over each derivative window. Returns dict(tu, {(part, window): derivative}).
    特徴量 F の時系列（アンテナ 1 本、試行全体で 1 回）。フレームごとに動いているビン・静止ビンで足し、50 Hz の格子に
    平均してから、Savitzky-Golay（2 次式）で微分の窓ごとに微分する"""
    from scipy.signal import savgol_filter
    S, t, v = d["S"], d["t"], d["v"]
    P = np.clip(S, 0, None)
    av = np.abs(v)
    st = av < static_v
    if st.all() or not st.any():
        raise RegError(f"静止ビンの境目 {static_v:g} m/s が速度軸（±{av.max():.2f} m/s）に合いません（--static_v）")
    p_mov = P[:, ~st].sum(axis=1)
    p_st = P[:, st].sum(axis=1)
    s1 = (P[:, ~st] * av[~st]).sum(axis=1)
    series = {"F1": np.log10(p_mov + EPS), "F2A": s1 / (p_st + EPS), "F2B": s1 / (p_mov + EPS)}
    dt = 1.0 / F_RATE
    tu = np.arange(t[0], t[-1] + 1e-9, dt)
    idx = np.clip(np.round((t - tu[0]) / dt).astype(int), 0, len(tu) - 1)
    cnt = np.bincount(idx, minlength=len(tu)).astype(float)
    have = cnt > 0
    out = {"tu": tu}
    for part, x in series.items():
        g = np.bincount(idx, weights=x, minlength=len(tu))[have] / cnt[have]
        g = np.interp(tu, tu[have], g)          # empty cells joined by lines / 空のマスは直線でつなぐ
        for w in deriv_windows:
            n = max(5, int(round(w / dt)) | 1)  # odd number of samples / サンプル数は奇数
            n = min(n, len(g) if len(g) % 2 else len(g) - 1)
            out[(part, w)] = savgol_filter(g, n, 2, deriv=1, delta=dt, mode="interp")
    return out


def trial_levels(d, a, b, static_v):
    """Signal levels of one antenna over [a, b): medians over the frames of the static power (|v| < static_v),
    the moving power, and the mean speed of the moving part. / 区間 [a, b) の強さ: 静止成分・動いている成分の強さと
    動いている成分の平均の速さ（フレームの中央値）"""
    sel = (d["t"] >= a) & (d["t"] < b)
    P = np.clip(d["S"][sel], 0, None)
    av = np.abs(d["v"])
    st = av < static_v
    p_st = P[:, st].sum(axis=1)
    p_mov = P[:, ~st].sum(axis=1)
    speed = (P[:, ~st] * av[~st]).sum(axis=1) / (p_mov + EPS)
    return dict(static=float(np.median(p_st)), moving=float(np.median(p_mov)), speed=float(np.median(speed)),
                rate=float(sel.sum() / max(b - a, 1e-9)))


def _log_levels(trials, out, out_dir):
    """Table of the signal levels of every trial, in dB from the first dataset's average.
    試行ごとの強さの表（最初のデータセットの平均からの dB）"""
    ds = lambda n: n.split("_")[0]   # noqa: E731
    ref_ds = ds(trials[0]["name"])
    ref = {}
    for a in (0, 1):
        for k in ("static", "moving"):
            ref[(a, k)] = np.mean([10 * np.log10(tr["levels"][a][k] + EPS) for tr in trials if ds(tr["name"]) == ref_ds])
    log(f"\n[試行ごとの強さ]（静止 = |v| が境目未満のビン、動 = それ以上のビンの、フレームごとの合計の中央値。"
        f"dB は {ref_ds} の平均から。速さ = 動いている成分の平均の速さの中央値）", out)
    log("  試行        ENMO平均   ant1 静止  ant1 動  ant1 動-静止  ant1 速さ   ant2 静止  ant2 動  ant2 動-静止  "
        "ant2 速さ  パケット/秒", out)
    rows = []
    for tr in trials:
        lv = tr["levels"]
        vals = []
        for a in (0, 1):
            st = 10 * np.log10(lv[a]["static"] + EPS) - ref[(a, "static")]
            mv = 10 * np.log10(lv[a]["moving"] + EPS) - ref[(a, "moving")]
            ratio = 10 * np.log10((lv[a]["moving"] + EPS) / (lv[a]["static"] + EPS))
            vals += [st, mv, ratio, lv[a]["speed"]]
        rate = lv[0]["rate"]
        rows.append([tr["name"], float(np.mean(tr["y"]))] + vals + [rate])
        log(f"  {tr['name']:10s} {np.mean(tr['y']):8.0f}   {vals[0]:+7.1f}  {vals[1]:+7.1f}  {vals[2]:+9.1f}    "
            f"{vals[3]:6.2f}    {vals[4]:+7.1f}  {vals[5]:+7.1f}  {vals[6]:+9.1f}    {vals[7]:6.2f}    {rate:6.0f}", out)
    with open(os.path.join(out_dir, "trial_levels.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["trial", "ENMO_mean(mG)"] + [f"ant{a}_{k}" for a in (1, 2) for k in
                                                  ("static(dB)", "moving(dB)", "moving_minus_static(dB)", "speed(m/s)")]
                   + ["frames_per_s"])
        for r in rows:
            w.writerow([r[0]] + [f"{x:.4g}" for x in r[1:]])


def deriv_features(tr, a, b):
    """Feature F of one window: the mean of |derivative| inside [a, b). / 区間の中の |微分| の平均"""
    m = (tr["tu"] >= a) & (tr["tu"] < b)
    if m.sum() < 2:
        return {k: np.nan for k in tr if k != "tu"}
    return {k: float(np.mean(np.abs(x[m]))) for k, x in tr.items() if k != "tu"}


D_GRID = (4, 10)   # feature D: coarse map of time x velocity cells / 特徴量 D の粗いマス（時間 x 速度）


def temporal_features(S):
    """Feature set D for one antenna and one window: how the Doppler changes inside the window.
    (1) the spread over time of each velocity bin's power (log10), (2) the Doppler map shrunk to
    4 time x 10 velocity cells (mean log10 power).
    特徴量 D（アンテナ 1 本・1 区間）: 区間の中の時間変化。(1) 速度ビンごとの強さの時間方向のばらつき、
    (2) ドップラーの図を時間 4 x 速度 10 のマスに縮めたもの"""
    L = np.log10(np.clip(S, EPS, None))
    tb = np.array_split(np.arange(L.shape[0]), D_GRID[0])
    vb = np.array_split(np.arange(L.shape[1]), D_GRID[1])
    grid = [L[np.ix_(ti, vi)].mean() for ti in tb for vi in vb]
    return np.concatenate([L.std(axis=0), grid])


def shape_spectrum(S):
    """Feature set E for one antenna and one window: the mean spectrum divided by its total (log10),
    i.e. only the shape of the speed distribution, not how strong the signal is.
    特徴量 E（アンテナ 1 本・1 区間）: 平均スペクトルを全体の強さで割ったもの（log10）。分布の形だけ"""
    m = np.clip(S, 0, None).mean(axis=0)
    return np.log10(np.clip(m / (m.sum() + EPS), EPS, None))


def cnn_input(S, t, a, b):
    """Window's Doppler resampled to a fixed number of frames (log10). / CNN の入力（時間を一定数にそろえる）"""
    n = int(min(320, round(CNN_FRAMES * (b - a))))
    tu = a + (np.arange(n) + 0.5) * (b - a) / n
    out = np.empty((n, S.shape[1]))
    for j in range(S.shape[1]):
        out[:, j] = np.interp(tu, t, S[:, j])
    return np.log10(np.clip(out, EPS, None))


def build_trial(spec, variant, window, out, need_c=True, need_ctx=(), need_d=False, need_e=False, column=TARGET,
                need_f=None, static_v=STATIC_V):
    """Windows of one trial: features A, B, C, CNN input and the target. / 1 試行分の区間"""
    name = spec["name"]
    files = spec["doppler"].get(variant, {})
    missing = [a for a in ("antenna1", "antenna2") if a not in files]
    if missing:
        raise RegError(f"{name}: {VARIANT_DIRS[variant]}/ に {', '.join(missing)} のドップラーがありません")
    ants = [load_doppler(files[a], out) for a in ("antenna1", "antenna2")]
    acc = [accel_traces(d) if need_c else None for d in ants]  # feature C only when used / C は使うときだけ
    der = [deriv_traces(d, *need_f) if need_f else None for d in ants]  # feature F only when used / F は使うときだけ
    XF = {}
    tgt = load_target(spec, window, column)
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
    XA, XB, XC, XD, XE, XN, y, kept = [], [], [], [], [], [], [], []
    need_ctx = sorted(set(need_ctx or ()))
    XNctx = {c: [] for c in need_ctx}
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
        fa, fb, fc, fd, fe, fn, ff = [], [], [], [], [], [], []
        fctx = {c: [] for c in need_ctx}
        ok = True
        for d, ac, de in zip(ants, acc, der):
            sel = (d["t"] >= a) & (d["t"] < b)
            if sel.sum() < 0.5 * rate * (b - a):  # too few frames (packet gap) / フレームが少なすぎる
                ok = False
                break
            half = max(2.5, (b - a) / 2)
            ctx = (d["t"] >= (a + b) / 2 - half) & (d["t"] < (a + b) / 2 + half)
            S = d["S"][sel]
            fa.append(np.log10(np.clip(S.mean(axis=0), EPS, None)))
            fb.append(physics_features(S, d["v"], d["t"][sel], d["t"][ctx], d["S"][ctx]))
            fc.append(accel_features(ac, a, b) if ac is not None else np.full(len(C_NAMES), np.nan))
            fd.append(temporal_features(S) if need_d else np.zeros(0))   # only when used / 使うときだけ
            fe.append(shape_spectrum(S) if need_e else np.zeros(0))
            if de is not None:
                ff.append(deriv_features(de, a, b))
            fn.append(cnn_input(d["S"], d["t"], a, b))
            for c in need_ctx:  # the window and c seconds before and after / 区間と前後 c 秒
                fctx[c].append(cnn_input(d["S"], d["t"], a - c, b + c))
        if not ok:
            dropped += 1
            continue
        XA.append(np.concatenate(fa))
        XB.append(np.concatenate(fb))
        XC.append(np.concatenate(fc))
        XD.append(np.concatenate(fd))
        XE.append(np.concatenate(fe))
        for key in (ff[0] if ff else {}):
            XF.setdefault(key, []).append([f_[key] for f_ in ff])
        XN.append(np.stack(fn, axis=-1).astype(np.float32))
        for c in need_ctx:
            XNctx[c].append(np.stack(fctx[c], axis=-1).astype(np.float32))
        y.append(yv)
        kept.append((a, b))
    if not y:
        raise RegError(f"{name}: ドップラーと IMU が重なる区間がありません (時間軸を確認してください)")
    log(f"  {name}: {len(y)} 区間（{window:g} 秒ごと、CSI の {kept[0][0]:.1f}〜{kept[-1][1]:.1f} s）"
        + (f"、{dropped} 区間はデータ不足で除外" if dropped else "")
        + f"  [{ants[0]['file']}, {ants[1]['file']}, {tgt['file']}]", out)
    return dict(name=name, XA=np.array(XA), XB=np.array(XB), XC=np.array(XC), XD=np.array(XD), XE=np.array(XE),
                XN=np.array(XN),
                XNctx={c: np.array(v) for c, v in XNctx.items()}, y=np.array(y),
                win=np.array(kept), v=ants[0]["v"], meta=ants[0]["meta"],
                levels=[trial_levels(d, kept[0][0], kept[-1][1], static_v) for d in ants],
                XF={k: np.array(v, float) for k, v in XF.items()})


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
    if kind == "svr":
        # RBF kernel; the target is standardized so epsilon and C do not depend on the mG scale
        # RBF カーネル。正解も標準化するので、epsilon と C は mG の大きさに依らない
        from sklearn.compose import TransformedTargetRegressor
        from sklearn.svm import SVR
        return TransformedTargetRegressor(
            regressor=make_pipeline(StandardScaler(), SVR(kernel="rbf", C=1.0, epsilon=0.1, gamma="scale")),
            transformer=StandardScaler())
    if kind == "gpr":
        # RBF + noise; kernel parameters fitted by the marginal likelihood on the training data
        # RBF + 雑音。カーネルのパラメータは学習データの周辺尤度で決める
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
        k = (ConstantKernel(1.0, (1e-2, 1e2)) * RBF(length_scale=10.0, length_scale_bounds=(1e-1, 1e4))
             + WhiteKernel(0.1, (1e-4, 1e1)))
        return make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=k, normalize_y=True,
                                                                        random_state=seed))
    if kind == "mlp":
        from sklearn.neural_network import MLPRegressor
        return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-2,
                                                            max_iter=3000, early_stopping=True,
                                                            validation_fraction=0.15, n_iter_no_change=50,
                                                            random_state=seed))
    raise ValueError(kind)


def _gain_jitter_layer(tf, scale):
    """Layer that, only while training, adds a random offset to the moving bins of each window:
    x + u * scale, u ~ U(-1, 1) drawn per window and per channel. scale (1, 1, bins, channels) is
    the jitter in the CNN's normalised input units (0 for the static bins and the mark channel).
    学習のときだけ、区間ごと・チャネルごとに動いている速度ビンへランダムな量を足す層（テストでは何もしない）"""
    class GainJitter(tf.keras.layers.Layer):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.scale = tf.constant(scale, tf.float32)

        def call(self, x, training=None):
            if not training:
                return x
            u = tf.random.uniform(tf.stack([tf.shape(x)[0], 1, 1, tf.shape(x)[-1]]), -1.0, 1.0)
            return x + u * self.scale
    return GainJitter(name="gain_jitter")


def _cnn_model(tf, shape, kind, jitter=None):
    """2D CNN on (frames x bins x channels), or 1D CNN along time on (frames x features).
    2 次元 CNN（時間 x 速度 x チャネル）、または時間方向だけの 1 次元 CNN（時間 x 特徴）"""
    L = tf.keras.layers
    inp = L.Input(shape=shape)
    x = inp
    if jitter is not None and kind != "1d":
        x = _gain_jitter_layer(tf, jitter)(x)
    if kind == "1d":
        for i, (n, k) in enumerate(((32, 5), (64, 5), (64, 3))):
            x = L.BatchNormalization()(L.Conv1D(n, k, padding="same")(x))
            x = L.ReLU(name="last_conv" if i == 2 else None)(x)
            if i < 2:
                x = L.MaxPooling1D(2)(x)
        x = L.GlobalAveragePooling1D()(x)
    else:
        for i, n in enumerate((16, 32, 64)):
            x = L.BatchNormalization()(L.Conv2D(n, 3, padding="same")(x))
            x = L.ReLU(name="last_conv" if i == 2 else None)(x)
            if i < 2:
                x = L.MaxPooling2D(2)(x)
        x = L.GlobalAveragePooling2D()(x)
    x = L.Dense(32, activation="relu")(L.Dropout(0.3)(x))
    return tf.keras.Model(inp, L.Dense(1)(x))


def jitter_mask(v, static_v, n_ch, n_ant=2):
    """Where --gain_jitter acts: the moving bins (|v| >= static_v) of the antenna channels, shape
    (1, 1, bins, channels); the mark channel of cnn3s / cnn5s is left alone.
    強さのゆらぎを掛ける場所: アンテナのチャネルの、動いている速度ビン（区間の印のチャネルには掛けない）"""
    m = np.zeros((1, 1, len(v), n_ch))
    m[0, 0, np.abs(np.asarray(v)) >= static_v, :min(n_ant, n_ch)] = 1.0
    return m


def fit_predict_cnn(Xtr, ytr, Xte, seed, epochs, kind="2d", cam=False, jitter_db=0.0, jitter_mask_=None):
    """Train a small CNN and predict. With cam=True, also the Grad-CAM map of each test window
    (where in time x velocity the CNN looked; the same size as the input, max 1).
    jitter_db > 0: while training, the moving bins (jitter_mask_) of each window and antenna are
    raised or lowered by a random gain of up to +-jitter_db dB (the input is log10 power, so +g dB
    = +g/10), so the CNN cannot rely on the absolute strength of the Doppler alone.
    小さな CNN を学習して予測する。cam=True なら、テストの区間ごとの Grad-CAM（CNN が時間 x 速度の
    どこを見たか。入力と同じ大きさ、最大 1）も返す。戻り値 (予測, Grad-CAM または None)。
    jitter_db > 0 なら、学習のときだけ区間ごと・アンテナごとに動いている成分の強さを最大 ±jitter_db dB
    ランダムに変える（入力は log10 なので +g dB は +g/10）。ドップラーの強さそのものに頼れなくする"""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    tf.keras.utils.set_random_seed(seed)
    ax = tuple(range(Xtr.ndim - 1))
    mu = Xtr.mean(axis=ax, keepdims=True)
    sd = Xtr.std(axis=ax, keepdims=True) + 1e-6
    ym, ys = ytr.mean(), ytr.std() + 1e-9
    jitter = None
    if jitter_db > 0 and jitter_mask_ is not None and kind != "1d":
        jitter = jitter_mask_ * (jitter_db / 10.0) / sd      # in normalised units / 正規化した入力の単位で
    model = _cnn_model(tf, Xtr.shape[1:], kind, jitter)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    cb = [tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)]
    model.fit(((Xtr - mu) / sd).astype(np.float32), (ytr - ym) / ys, validation_split=0.15, epochs=epochs,
              batch_size=16, verbose=0, callbacks=cb, shuffle=True)
    Xn = ((Xte - mu) / sd).astype(np.float32)
    pred = model.predict(Xn, verbose=0).ravel() * ys + ym
    if not cam or kind == "1d":
        return pred, None
    gm = tf.keras.Model(model.input, [model.get_layer("last_conv").output, model.output])
    maps = []
    for i in range(0, len(Xn), 64):
        xb = tf.convert_to_tensor(Xn[i:i + 64])
        with tf.GradientTape() as tape:
            f, yb = gm(xb, training=False)
        g = tape.gradient(yb, f)
        w = tf.reduce_mean(g, axis=(1, 2), keepdims=True)                  # weight of each feature map / 重み
        m = tf.nn.relu(tf.reduce_sum(w * f, axis=-1))[..., None]            # where it raises ENMO / ENMO を上げる場所
        m = tf.image.resize(m, Xte.shape[1:3])[..., 0].numpy()
        maps.append(m / (m.reshape(len(m), -1).max(axis=1)[:, None, None] + 1e-12))
    return pred, np.concatenate(maps).astype(np.float32)


def metrics(y, p, mape_min=None):
    """MAE, RMSE, R2, r and MAPE [%] (MAPE only over windows with |truth| >= mape_min; MAPE_n = their count).
    MAE・RMSE・R²・r と MAPE [%]（MAPE は正解が mape_min 以上の区間だけ。MAPE_n はその区間の数）"""
    y, p = np.asarray(y, float), np.asarray(p, float)
    err = p - y
    ss = np.sum((y - y.mean()) ** 2)
    r = float(np.corrcoef(y, p)[0, 1]) if np.std(p) > 0 and np.std(y) > 0 else float("nan")
    lo = MAPE_MIN if mape_min is None else mape_min
    m = np.abs(y) >= max(lo, EPS)
    mape = float(np.mean(np.abs(err[m]) / np.abs(y[m])) * 100) if m.any() else float("nan")
    return dict(MAE=float(np.mean(np.abs(err))), RMSE=float(np.sqrt(np.mean(err ** 2))),
                R2=float(1 - np.sum(err ** 2) / ss) if ss > 0 else float("nan"), r=r, MAPE=mape, MAPE_n=int(m.sum()))


# ============================================================ run
def parse_splits(items):
    """--split "A1:A2" / "A1,A2_XR1:A2_XR2" / "cv" -> [("cv", None, None) | (label, train tokens, test tokens)].
    学習とテストの分け方。左が学習、右がテスト（データセット名・試行名・A2_* のような書き方、カンマ区切り）"""
    if not items:
        return [("cv", None, None)]
    out = []
    for it in items:
        it = it.strip()
        if it.lower() == "cv":
            out.append(("cv", None, None))
            continue
        if it.count(":") != 1:
            raise RegError(f"--split は「学習:テスト」の形で書きます（例 \"A1:A2\"、\"A1,A2_XR1:A2_XR2\"）: {it}")
        a, b = it.split(":")
        tr = [x.strip() for x in a.split(",") if x.strip()]
        te = [x.strip() for x in b.split(",") if x.strip()]
        if not tr or not te:
            raise RegError(f"--split の学習とテストの両方を書いてください: {it}")
        out.append((f"学習 {'+'.join(tr)} → テスト {'+'.join(te)}", tr, te))
    return out


def _pick_trials(tokens, specs):
    """Trial names matched by dataset names, trial names or patterns (A2_*). / データセット名・試行名・パターンで選ぶ"""
    import fnmatch
    names = set()
    for tok in tokens:
        hit = {sp["name"] for sp in specs
               if tok.lower() in (sp["name"].lower(), (sp.get("ds") or "").lower())
               or fnmatch.fnmatch(sp["name"].lower(), tok.lower())}
        if not hit:
            dss = sorted({sp.get("ds") for sp in specs if sp.get("ds")})
            raise RegError(f"--split の「{tok}」に当たる試行がありません（データセット: {', '.join(dss) or 'なし'}、"
                           f"試行: {', '.join(sp['name'] for sp in specs)}）")
        names |= hit
    return names


def make_evaluations(splits, trials, specs, tname, blocks, out):
    """[(label, [(train mask, test mask), ...])] for each requested split. / 分け方ごとの学習・テストの区間"""
    evals = []
    for kind, tr_tok, te_tok in splits:
        if kind == "cv":
            if len(trials) >= 2:
                g = np.concatenate([np.full(len(tr["y"]), k) for k, tr in enumerate(trials)])
                how = f"試行ごと（{len(trials)} 試行。1 試行をテスト、残りで学習）"
            else:
                n = len(trials[0]["y"])
                g = np.minimum((np.arange(n) * blocks) // n, blocks - 1)
                how = f"試行が 1 本なので、時間を {blocks} つのかたまりに分けて交互にテスト（参考値）"
            log(f"  評価: {how}", out)
            evals.append(("試行ごと", [(g != k, g == k) for k in np.unique(g)]))
            continue
        tr_names, te_names = _pick_trials(tr_tok, specs), _pick_trials(te_tok, specs)
        both = tr_names & te_names
        if both:
            raise RegError(f"学習とテストの両方に入っている試行があります: {', '.join(sorted(both))}（{kind}）")
        tr_m, te_m = np.isin(tname, sorted(tr_names)), np.isin(tname, sorted(te_names))
        log(f"  評価: {kind}（学習 {len(tr_names)} 試行・{int(tr_m.sum())} 区間: {', '.join(sorted(tr_names))} ／ "
            f"テスト {len(te_names)} 試行・{int(te_m.sum())} 区間: {', '.join(sorted(te_names))}）", out)
        evals.append((kind, [(tr_m, te_m)]))
    return evals


def run(opts):
    global MAPE_MIN
    out = []
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = opts.out or os.path.join(HERE, "outputs", f"regression_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    log(f"=== ドップラー → ENMO の回帰 ({stamp}) ===", out)
    if not os.path.isdir(opts.input):
        raise RegError(f"入力フォルダがありません: {opts.input}")
    specs = discover(opts.input, out, getattr(opts, "pairs", None), getattr(opts, "exclude", EXCLUDE_DEFAULT))
    target = getattr(opts, "target", "waist") or "waist"
    if target not in TARGETS:
        raise RegError(f"--target は {' / '.join(TARGETS)} です（{target}）")
    column = TARGETS[target]
    log(f"正解: {TARGET_JA[target]}（列 {column}）", out)
    MAPE_MIN = float(getattr(opts, "mape_min", None) if getattr(opts, "mape_min", None) is not None else 50.0)
    log(f"MAPE（平均絶対パーセント誤差）: 正解が {MAPE_MIN:g} mG 未満の区間は除いて計算（静止で ENMO ≈ 0 になり割り算が発散するため）", out)
    if target != "waist":
        # trials without the whole-body ENMO (fewer than four sensors) are left out / 全身の ENMO がない試行は使わない
        lack = [sp["name"] for sp in specs if not has_target(sp, opts.windows[0], column)]
        if lack:
            log(f"  注意: 全身の ENMO がない試行は使いません（センサが 4 台そろっていない、または加速度計算ソフトが古い）: "
                f"{', '.join(lack)}", out)
            specs = [sp for sp in specs if sp["name"] not in lack]
        if not specs:
            raise RegError("全身の ENMO（ENMO_body(mG)）がある試行がありません。4 台そろった試行を加速度計算ソフト v2.0 で"
                           "計算し直してください")
    variants = list(VARIANT_DIRS) if opts.doppler == "all" else [opts.doppler]
    models = [m.strip() for m in opts.models.split(",") if m.strip()]
    bad = [m for m in models if m not in ALL_MODELS + STUDY_MODELS + EXTRA_MODELS]
    if bad:
        raise RegError(f"不明なモデル: {', '.join(bad)}（使えるのは {', '.join(ALL_MODELS + STUDY_MODELS + EXTRA_MODELS)}）")
    study = bool(getattr(opts, "cnn_study", False))
    if study:  # the CNN and its check variants / CNN と、理由を確かめるための CNN
        models += [m for m in ["cnn"] + STUDY_MODELS if m not in models]
    n_seeds = max(1, int(getattr(opts, "cnn_seeds", 1) or 1))
    feats = [f.strip().upper().replace(" ", "") for f in opts.features.split(",") if f.strip()]
    def _ok(f):
        base, _, k = f.partition("@")
        return bool(base) and all(p in ("A", "B", "C", "D", "E") + F_PARTS for p in base.split("+")) and (
            k == "" or (k.isdigit() and 1 <= int(k) <= 5))
    badf = [f for f in feats if not _ok(f)]
    if badf:
        raise RegError(f"不明な特徴量: {', '.join(badf)}（A, B, C, D, E, F1, F2a, F2b と、その組み合わせ A+D など。"
                       "前後の区間も入れるときは B+D@1 のように @ と区間の数 1〜5）")
    used = {p for f in feats for p in f.partition("@")[0].split("+")}
    static_v = float(getattr(opts, "static_v", None) or STATIC_V)
    dws = getattr(opts, "deriv_window", None) or [DERIV_WINDOW]
    dws = parse_windows(dws) if isinstance(dws, str) else [float(w) for w in dws]
    need_f = (static_v, dws) if used & set(F_PARTS) else None
    if need_f:
        log(f"特徴量 F: 静止ビン |v| < {static_v:g} m/s を除き、{F_RATE:g} Hz にそろえて Savitzky-Golay で微分"
            f"（微分の窓 {', '.join(f'{w:g}' for w in dws)} 秒）、1 秒ごとに |微分| を平均。"
            + "、".join(f"{p}: {F_DESC[p]}" for p in F_PARTS if p in used), out)
        if len(dws) > 1:   # one set of results per derivative window / 微分の窓ごとに別の結果
            feats = [f if not set(f.partition("@")[0].split("+")) & set(F_PARTS) else
                     f"{f.partition('@')[0]}/{w:g}s" + (f"@{f.partition('@')[2]}" if "@" in f else "")
                     for f in feats for w in (dws if set(f.partition("@")[0].split("+")) & set(F_PARTS) else [None])]
    jitters = parse_jitter(getattr(opts, "gain_jitter", None) or "0")
    if any(j > 0 for j in jitters) and any(m in CNN_KINDS for m in models):
        log(f"CNN の学習で強さをゆらす: 区間ごと・アンテナごとに、動いている速度ビン（|v| ≥ {static_v:g} m/s）の強さを"
            f"最大 ±{'/'.join(f'{j:g}' for j in jitters if j > 0)} dB ランダムに変える（テストの入力はそのまま。"
            "1 次元 CNN には掛けない）" + ("。0 dB（ゆらさない）とも比べる" if 0.0 in jitters else ""), out)
    splits = parse_splits(getattr(opts, "split", None))
    for kind, tr_tok, te_tok in splits:  # check the names before the slow part / 時間の掛かる処理の前に名前を確かめる
        if tr_tok:
            both = _pick_trials(tr_tok, specs) & _pick_trials(te_tok, specs)
            if both:
                raise RegError(f"学習とテストの両方に入っている試行があります: {', '.join(sorted(both))}（{kind}）")
    results, pred_rows = [], []
    windows = opts.windows
    for variant, window in [(v, w) for v in variants for w in windows]:
        log(f"\n[ドップラー: {variant}（{VARIANT_DIRS[variant]}/）、区間 {window:g} 秒]", out)
        try:
            trials = [build_trial(sp, variant, window, out, need_c="C" in used, need_d="D" in used, need_e="E" in used,
                                  need_ctx=[CONTEXT_S[m] for m in models if m in CONTEXT_S], column=column,
                                  need_f=need_f, static_v=static_v)
                      for sp in specs]
        except RegError as e:
            if opts.doppler == "all":
                log(f"  この版は飛ばします: {e}", out)
                continue
            raise
        _log_levels(trials, out, out_dir)
        if getattr(opts, "levels_only", False):
            continue
        XA = np.concatenate([tr["XA"] for tr in trials])
        XB = np.concatenate([tr["XB"] for tr in trials])
        XC = np.concatenate([tr["XC"] for tr in trials])
        XN = np.concatenate([tr["XN"] for tr in trials])
        cnn_cache = {}

        def cnn_data(kind):
            """Input of each CNN variant (made once). / CNN の種類ごとの入力（1 回だけ作る）"""
            if kind not in cnn_cache:
                if kind == "cnn":
                    cnn_cache[kind] = (XN, "2d")
                elif kind == "cnn_shuffle":  # frames in a random order in each window / 区間の中で時間の順番を入れ替え
                    rng = np.random.default_rng(1234)
                    cnn_cache[kind] = (np.stack([x[rng.permutation(len(x))] for x in XN]), "2d")
                elif kind == "cnn_timemean":  # every frame = the window's mean / どの時刻も区間の平均
                    cnn_cache[kind] = (np.repeat(XN.mean(axis=1, keepdims=True), XN.shape[1], axis=1), "2d")
                elif kind == "cnn1d":  # time x (velocity bins x antennas) / 時間 x (速度 x アンテナ)
                    cnn_cache[kind] = (XN.reshape(len(XN), XN.shape[1], -1), "1d")
                elif kind in CONTEXT_S:  # window +- c s, and a channel marking the window / 区間と前後 c 秒 + 区間の印
                    c = CONTEXT_S[kind]
                    X3 = np.concatenate([tr["XNctx"][c] for tr in trials])
                    nf = X3.shape[1]
                    i0 = int(round(nf * c / (window + 2 * c)))
                    i1 = int(round(nf * (c + window) / (window + 2 * c)))
                    mark = np.zeros((1, nf, X3.shape[2], 1), np.float32)
                    mark[:, i0:i1] = 1.0
                    cnn_cache[kind] = (np.concatenate([X3, np.broadcast_to(mark, X3.shape[:3] + (1,))], axis=-1), "2d")
            return cnn_cache[kind]
        XD = np.concatenate([tr["XD"] for tr in trials])
        XE = np.concatenate([tr["XE"] for tr in trials])
        fsets = {"A": XA, "B": XB, "C": XC, "D": XD, "E": XE}
        fnames = feature_names(trials[0]["v"])
        XF = {}
        if need_f:   # feature F, one set per derivative window / 特徴量 F（微分の窓ごと）
            for key in trials[0]["XF"]:
                XF[key] = np.concatenate([tr["XF"][key] for tr in trials])
                part, w = key
                fsets[(part, w)] = XF[key]
                fnames[(part, w)] = [f"{part}{f'_w{w:g}' if len(dws) > 1 else ''}_ant{a}" for a in (1, 2)]
        x_cache = {}

        def tab_data(feat):
            """Input of a tabular model and its column names (made once). / 表の形のモデルの入力と列の名前"""
            if feat not in x_cache:
                base, _, k = feat.partition("@")
                base, _, wtag = base.partition("/")
                w = float(wtag[:-1]) if wtag else (dws[0] if need_f else None)
                keys = [(p, w) if p in F_PARTS else p for p in base.split("+")]
                X = np.hstack([fsets[q] for q in keys])
                names = [n for q in keys for n in fnames[q]]
                if k:
                    X, names = add_context(X, names, tname, int(k))
                x_cache[feat] = (X, names)
            return x_cache[feat]
        y = np.concatenate([tr["y"] for tr in trials])
        tname = np.concatenate([[tr["name"]] * len(tr["y"]) for tr in trials])
        win = np.concatenate([tr["win"] for tr in trials])
        evals = make_evaluations(splits, trials, specs, tname, opts.blocks, out)
        _dump_features(os.path.join(out_dir, f"features_{variant}_{window:g}s.csv"), tname, win, XA, XB, XC, y,
                       trials[0]["v"], XD, XE, column=column, XF={k: (XF[k], fnames[k]) for k in XF})
        if need_f:
            _static_check(out_dir, f"{variant}_{window:g}s", XA, y, trials[0]["v"], static_v, trials[0]["meta"], out,
                          plots=not opts.no_plots)
        configs = [("baseline", "-", 0.0)]
        for m in models:
            if m in CNN_KINDS:
                configs += [(m, CNN_FEAT[m] + (f" 強さ±{j:g}dB" if j > 0 else ""), j) for j in jitters
                            if not (j > 0 and m == "cnn1d")]
            else:
                configs += [(m, f, 0.0) for f in feats]
        for sp_label, folds in evals:
            if len(evals) > 1 or sp_label != "試行ごと":
                log(f"  -- 学習とテスト: {sp_label}", out)
            for model, feat, jit in configs:
                t_start = time.time()
                seeds = [opts.seed + k for k in range(n_seeds)] if model in CNN_KINDS else [opts.seed]
                preds, skip = [], False
                gp_sd = np.full(len(y), np.nan) if model == "gpr" else None
                cams = None
                want_imp = bool(getattr(opts, "importance", False))
                imp_sum, imp_n = None, 0
                for si, seed in enumerate(seeds):
                    pred = np.full(len(y), np.nan)
                    for tr_i, te_i in folds:
                        if model == "baseline":
                            pred[te_i] = y[tr_i].mean()
                        elif model in CNN_KINDS:
                            Xin, ck = cnn_data(model)
                            want_cam = study and model == "cnn" and si == 0 and not opts.no_plots
                            try:
                                jm = jitter_mask(trials[0]["v"], static_v, Xin.shape[-1]) if jit > 0 else None
                                p_, cm = fit_predict_cnn(Xin[tr_i], y[tr_i], Xin[te_i], seed, opts.epochs, kind=ck,
                                                         cam=want_cam, jitter_db=jit, jitter_mask_=jm)
                            except ImportError:
                                log("  CNN: TensorFlow がないので飛ばします (pip install tensorflow)", out)
                                skip = True
                                break
                            pred[te_i] = p_
                            if cm is not None:
                                if cams is None:
                                    cams = np.full((len(y),) + cm.shape[1:], np.nan, np.float32)
                                cams[te_i] = cm
                        else:
                            X, xnames = tab_data(feat)
                            est = make_tabular(model, seed)
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")  # GPR / SVR convergence notes / 収束の注意は出さない
                                est.fit(X[tr_i], y[tr_i])
                                if model == "gpr":
                                    pred[te_i], gp_sd[te_i] = est.predict(X[te_i], return_std=True)
                                else:
                                    pred[te_i] = est.predict(X[te_i])
                            if want_imp and model in ("rf", "xgboost"):
                                imp_sum = est.feature_importances_ + (0 if imp_sum is None else imp_sum)
                                imp_n += 1
                    if skip:
                        break
                    preds.append(pred)
                if skip or not preds:
                    continue
                pred = np.mean(preds, axis=0)  # several seeds: their average / 複数回なら予測の平均
                te = np.any([f[1] for f in folds], axis=0)   # the test windows of this evaluation / テストの区間
                if np.isnan(pred[te]).all():
                    continue
                yy, pp, tn_, ww = y[te], pred[te], tname[te], win[te]
                mt = metrics(yy, pp)
                seed_m = [metrics(yy, p_[te]) for p_ in preds] if len(preds) > 1 else []
                extra = dict(n_seeds=len(preds), seed_metrics=seed_m,
                             MAE_sd=float(np.std([m_["MAE"] for m_ in seed_m])) if seed_m else float("nan"),
                             R2_sd=float(np.std([m_["R2"] for m_ in seed_m])) if seed_m else float("nan"),
                             MAPE_sd=float(np.std([m_["MAPE"] for m_ in seed_m])) if seed_m else float("nan"))
                per = {tn: metrics(yy[tn_ == tn], pp[tn_ == tn]) for tn in dict.fromkeys(tn_)}
                label = "平均値で予測（基準）" if model == "baseline" else MODEL_NAMES[model]
                results.append(dict(variant=variant, window=window, split=sp_label, model=model, label=label,
                                    features=feat, jitter=jit, **mt, per_trial=per, seconds=time.time() - t_start,
                                    pred=pp, y=yy, tname=tn_, win=ww, target=target, **extra))
                if imp_n:
                    imp = imp_sum / imp_n
                    results[-1]["importance"] = (tab_data(feat)[1], imp / max(imp.sum(), 1e-12))
                log(f"  {label:16s} 特徴量 {feat:10s}: MAE {mt['MAE']:7.2f} mG  MAPE {mt['MAPE']:6.1f} %  RMSE {mt['RMSE']:7.2f}  "
                    f"R² {mt['R2']:6.3f}  r {mt['r']:6.3f}  ({time.time() - t_start:.0f} s)", out)
                if seed_m:
                    each = ", ".join("%.2f" % m_["R2"] for m_ in seed_m)
                    log(f"      乱数の種 {len(seed_m)} 回: R² {np.mean([m_['R2'] for m_ in seed_m]):.3f} ± {extra['R2_sd']:.3f}"
                        f"（{each}）、MAE {np.mean([m_['MAE'] for m_ in seed_m]):.1f}"
                        f" ± {extra['MAE_sd']:.1f} mG、MAPE {np.mean([m_['MAPE'] for m_ in seed_m]):.1f} ± {extra['MAPE_sd']:.1f} %。"
                        f"上の値は {len(seed_m)} 回の予測の平均", out)
                if gp_sd is not None:
                    sdv = gp_sd[te]
                    cover = float(np.mean(np.abs(yy - pp) <= 1.96 * sdv))
                    log(f"      ガウス過程の予測の幅（±1.96σ、平均 ±{1.96 * np.mean(sdv):.0f} mG）に正解が入った割合 "
                        f"{cover:.0%}（95 % に近いほど幅が正しい）", out)
                if cams is not None:
                    _plot_gradcam(out_dir, f"{_split_tag(sp_label)}{variant}_{window:g}s", XN[te], cams[te], yy,
                                  trials[0]["v"], window, sp_label, out)
                for i in range(len(yy)):
                    pred_rows.append([variant, f"{window:g}", sp_label, model, feat, tn_[i], f"{ww[i][0]:.3f}",
                                      f"{ww[i][1]:.3f}", f"{yy[i]:.3f}", f"{pp[i]:.3f}"])
    if getattr(opts, "levels_only", False):   # only the level table / 強さの表だけ
        with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(vars(opts), f, ensure_ascii=False, indent=2)
        log(f"\n保存しました: {out_dir}（trial_levels.csv）", out)
        with open(os.path.join(out_dir, "log.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(out) + "\n")
        return [], out_dir
    if not results:
        raise RegError("どの版でも計算できませんでした")
    _save(out_dir, results, pred_rows, opts, out)
    if any("importance" in r for r in results):
        _save_importance(out_dir, results, out, plots=not opts.no_plots)
    if not opts.no_plots:
        _plots(out_dir, results)
    log(f"\n保存しました: {out_dir}", out)
    with open(os.path.join(out_dir, "log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    return results, out_dir


def d_feature_names(v):
    """Column names of feature set D (per antenna: std of each bin, then the coarse grid). / 特徴量 D の列の名前"""
    vb = np.array_split(np.arange(len(v)), D_GRID[1])
    names = []
    for a in (1, 2):
        names += [f"D_ant{a}_std_v{x:+.3f}" for x in v]
        names += [f"D_ant{a}_grid_t{i + 1}_v{v[b_[0]]:+.1f}to{v[b_[-1]]:+.1f}" for i in range(D_GRID[0]) for b_ in vb]
    return names


def feature_names(v):
    """Column names of every feature set. / 特徴量ごとの列の名前"""
    return {"A": [f"A_ant{a}_v{x:+.3f}" for a in (1, 2) for x in v], "B": B_NAMES_full(),
            "C": [f"C_ant{a}_{n}" for a in (1, 2) for n in C_NAMES], "D": d_feature_names(v),
            "E": [f"E_ant{a}_v{x:+.3f}" for a in (1, 2) for x in v]}


def add_context(X, names, tname, k):
    """Append the features of the k windows before and after (same trial; at the ends the nearest window).
    前後 k 区間の特徴量を横に足す（同じ試行の中。端では一番近い区間）"""
    out = [X]
    names = list(names)
    offsets = [o for o in range(-k, k + 1) if o != 0]
    for o in offsets:
        Z = np.empty_like(X)
        for tn in dict.fromkeys(tname):
            idx = np.where(tname == tn)[0]
            Z[idx] = X[idx[np.clip(np.arange(len(idx)) + o, 0, len(idx) - 1)]]
        out.append(Z)
    names_all = names + [f"t{o:+d}:{n}" for o in offsets for n in names]
    return np.hstack(out), names_all


def _feature_group(name):
    """Group of a column for the summary (t-1: ... / B_mean_speed / D_std / ...). / まとめ用のグループ"""
    pre, _, n = name.rpartition(":")
    if n.startswith("B_"):
        g = "B_" + n.split("_", 2)[2]            # B_ant1_mean_speed -> B_mean_speed
    elif n.startswith("D_"):
        g = "D_" + n.split("_")[2]               # D_ant1_std_v.. -> D_std / D_grid
    else:
        g = n.split("_")[0]                      # A / C / E
    return (pre + ":" if pre else "") + g


def _save_importance(out_dir, results, out, plots=True, top=20):
    """Feature importance of the tree models (mean over the folds): a table of all, the top for the best model
    of each split (log and figure). / 決定木のモデルの特徴量の重要度（学習した回の平均）"""
    rows, groups = [], []
    for r in results:
        if "importance" not in r:
            continue
        names, imp = r["importance"]
        order = np.argsort(-imp)
        for rank, j in enumerate(order, 1):
            rows.append([r["split"], r["label"], r["features"], rank, names[j], f"{imp[j]:.5f}"])
        g = {}
        for n, x in zip(names, imp):
            g[_feature_group(n)] = g.get(_feature_group(n), 0.0) + float(x)
        for k_, x in sorted(g.items(), key=lambda kv: -kv[1]):
            groups.append([r["split"], r["label"], r["features"], k_, f"{x:.4f}"])
    with open(os.path.join(out_dir, "importance.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "model", "features", "rank", "feature", "importance"])
        w.writerows(rows)
    with open(os.path.join(out_dir, "importance_groups.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "model", "features", "group", "importance"])
        w.writerows(groups)
    log("\n特徴量の重要度（決定木のモデル。importance.csv・importance_groups.csv）", out)
    for sp in dict.fromkeys(r["split"] for r in results):
        cand = [r for r in results if r["split"] == sp and "importance" in r]
        if not cand:
            continue
        b = max(cand, key=lambda r: r["R2"])
        names, imp = b["importance"]
        order = np.argsort(-imp)[:top]
        g = {}
        for n, x in zip(names, imp):
            g[_feature_group(n)] = g.get(_feature_group(n), 0.0) + float(x)
        gtxt = "、".join(f"{k_} {x:.0%}" for k_, x in sorted(g.items(), key=lambda kv: -kv[1])[:6])
        log(f"  [{sp}] {b['label']}（特徴量 {b['features']}、R² {b['R2']:.3f}）: 上位 "
            + "、".join(f"{names[j]} {imp[j]:.1%}" for j in order[:5]) + f"／グループ別 {gtxt}", out)
        if not plots:
            continue
        try:
            plt = _setup_plt()
            fig, ax = plt.subplots(figsize=(9, 0.32 * len(order) + 1.4))
            ax.barh(np.arange(len(order))[::-1], imp[order], color="#3A9D5D")
            ax.set_yticks(np.arange(len(order))[::-1])
            ax.set_yticklabels([names[j] for j in order], fontsize=8)
            ax.set_xlabel("重要度（合計 1）")
            ax.set_title(f"{b['label']}（特徴量 {b['features']}、R² {b['R2']:.2f}）上位 {len(order)} 個\n{sp}",
                         fontsize=10)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"importance_{_split_tag(sp) or 'cv_'}{b['model']}.png"), dpi=110)
            plt.close(fig)
        except Exception as e:  # the figure is optional / 図は失敗しても続ける
            log(f"  注意: 重要度の図を作れませんでした（{e}）", out)


def _dump_features(path, tname, win, XA, XB, XC, y, v, XD=None, XE=None, column=TARGET, XF=None):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        na = XA.shape[1] // 2
        has_c = bool(np.isfinite(XC).any())  # feature C only when it was computed / C は計算したときだけ
        has_d = XD is not None and XD.ndim == 2 and XD.shape[1] > 0
        has_e = XE is not None and XE.ndim == 2 and XE.shape[1] > 0
        d_names = d_feature_names(v) if has_d else []
        w.writerow(["trial", "start_csi(s)", "end_csi(s)", column] + B_NAMES_full()
                   + ([f"C_ant{a}_{n}" for a in (1, 2) for n in C_NAMES] if has_c else [])
                   + d_names + ([f"E_ant{a}_v{x:+.3f}" for a in (1, 2) for x in v] if has_e else [])
                   + [n for _, ns in (XF or {}).values() for n in ns]
                   + [f"A_ant{1 + j // na}_v{v[j % na]:+.3f}" for j in range(XA.shape[1])])
        for i in range(len(y)):
            w.writerow([tname[i], f"{win[i][0]:.3f}", f"{win[i][1]:.3f}", f"{y[i]:.3f}"]
                       + [f"{x:.6g}" for x in XB[i]] + ([f"{x:.6g}" for x in XC[i]] if has_c else [])
                       + ([f"{x:.6g}" for x in XD[i]] if has_d else []) + ([f"{x:.6g}" for x in XE[i]] if has_e else [])
                       + [f"{x:.6g}" for X_, _ in (XF or {}).values() for x in X_[i]]
                       + [f"{x:.6g}" for x in XA[i]])


def B_NAMES_full():
    return [f"B_ant{a}_{n}" for a in (1, 2) for n in B_NAMES]


def _save(out_dir, results, pred_rows, opts, out):
    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doppler", "window(s)", "split", "model", "features", "MAE(mG)", "RMSE(mG)", "R2", "r", "seconds",
                    "n_seeds", "MAE_sd(mG)", "R2_sd", "target", "MAPE(%)", "MAPE_sd(%)", "MAPE_windows", "MAPE_min(mG)"])
        for r in results:
            sd_ = r.get("seed_metrics")
            w.writerow([r["variant"], f"{r['window']:g}", r["split"], r["label"], r["features"], f"{r['MAE']:.3f}",
                        f"{r['RMSE']:.3f}", f"{r['R2']:.4f}", f"{r['r']:.4f}", f"{r['seconds']:.1f}",
                        r.get("n_seeds", 1), f"{r['MAE_sd']:.3f}" if sd_ else "", f"{r['R2_sd']:.4f}" if sd_ else "",
                        r.get("target", "waist"), f"{r['MAPE']:.2f}", f"{r['MAPE_sd']:.2f}" if sd_ else "",
                        r.get("MAPE_n", ""), f"{MAPE_MIN:g}"])
    if any(r.get("seed_metrics") for r in results):
        with open(os.path.join(out_dir, "cnn_seeds.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["doppler", "window(s)", "split", "model", "features", "seed", "MAE(mG)", "RMSE(mG)", "R2", "r",
                        "MAPE(%)"])
            for r in results:
                for k, m_ in enumerate(r.get("seed_metrics") or []):
                    w.writerow([r["variant"], f"{r['window']:g}", r["split"], r["label"], r["features"], opts.seed + k,
                                f"{m_['MAE']:.3f}", f"{m_['RMSE']:.3f}", f"{m_['R2']:.4f}", f"{m_['r']:.4f}",
                                f"{m_['MAPE']:.2f}"])
    with open(os.path.join(out_dir, "metrics_by_trial.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doppler", "window(s)", "split", "model", "features", "trial", "MAE(mG)", "RMSE(mG)", "R2", "r",
                    "MAPE(%)"])
        for r in results:
            for tn, m in r["per_trial"].items():
                w.writerow([r["variant"], f"{r['window']:g}", r["split"], r["label"], r["features"], tn, f"{m['MAE']:.3f}",
                            f"{m['RMSE']:.3f}", f"{m['R2']:.4f}", f"{m['r']:.4f}", f"{m['MAPE']:.2f}"])
    with open(os.path.join(out_dir, "predictions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doppler", "window(s)", "split", "model", "features", "trial", "start_csi(s)", "end_csi(s)",
                    "true_ENMO(mG)", "pred_ENMO(mG)"])
        w.writerows(pred_rows)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(opts), f, ensure_ascii=False, indent=2)
    for sp, wl in dict.fromkeys((r["split"], r["window"]) for r in results):
        best = min((r for r in results if r["model"] != "baseline" and r["window"] == wl and r["split"] == sp),
                   key=lambda r: r["MAE"], default=None)
        if best:
            log(f"\n[{sp}] {wl:g} 秒の区間で一番誤差が小さかったのは {best['label']}（特徴量 {best['features']}、"
                f"ドップラー {best['variant']}）: MAE {best['MAE']:.2f} mG、MAPE {best['MAPE']:.1f} %、R² {best['R2']:.3f}", out)


# ============================================================ plots
def _static_check(out_dir, tag, XA, y, v, static_v, meta, out, plots=True, drop_db=20.0):
    """How wide the static path is in the real data: the mean spectrum of the quietest windows (lowest 5 % of the
    ENMO), in dB from its peak, and the smallest |v| where it has dropped by drop_db on both sides, next to the
    Hann window's main lobe computed from the Doppler settings. Saves static_spectrum_<tag>.csv (and .png).
    静止成分が実データでどれだけ広がっているか: ENMO が一番小さい 5 % の区間の平均スペクトル（ピークからの dB）と、
    両側で drop_db 下がる |v|。ドップラーの設定から計算したハン窓の主ローブと並べる"""
    n = XA.shape[1] // 2
    k = max(3, int(round(0.05 * len(y))))
    quiet = np.argsort(y)[:k]
    rows, edge = {}, {}
    for a in (0, 1):
        spec = np.mean(10 ** XA[quiet, a * n:(a + 1) * n], axis=0)
        db = 10 * np.log10(spec / spec.max())
        rows[a] = db
        i0 = int(np.argmax(spec))
        sides = []
        for step in (1, -1):
            i = i0
            while 0 <= i + step < n and db[i] > -drop_db:
                i += step
            sides.append(abs(v[i]))
        edge[a] = max(sides)
    theory = None
    N, res = meta.get("sample_length"), meta.get("delta_v_resolution")
    if N and res:   # Hann window of N packets: power response vs velocity / N パケットのハン窓の応答
        f = np.abs(v) / (float(res) * float(N))   # cycles per packet / 1 パケットあたりの周期
        from scipy.signal.windows import hann   # the same window as CSI_doppler_computation.py / ドップラー計算と同じ窓
        nn = np.arange(int(N))
        w = hann(int(N))
        resp = np.abs(np.array([np.sum(w * np.exp(-2j * np.pi * fi * nn)) for fi in f])) ** 2
        theory = 10 * np.log10(resp / resp.max() + 1e-30)
    path = os.path.join(out_dir, f"static_spectrum_{tag}.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w_ = csv.writer(fh)
        w_.writerow(["v(m/s)", "antenna1(dB)", "antenna2(dB)"] + (["hann_main_lobe(dB)"] if theory is not None else []))
        for i in range(n):
            w_.writerow([f"{v[i]:+.4f}", f"{rows[0][i]:.2f}", f"{rows[1][i]:.2f}"]
                        + ([f"{theory[i]:.2f}"] if theory is not None else []))
    log(f"  静止成分の広がり（ENMO が小さい {k} 区間、平均 {np.mean(y[quiet]):.0f} mG）: 0 m/s の山から {drop_db:g} dB 下がるのは "
        f"|v| ≥ {edge[0]:.2f} m/s（antenna1）、{edge[1]:.2f} m/s（antenna2）。今の境目 {static_v:g} m/s"
        + (f"（ハン窓 {int(N)} パケットの主ローブの端 {2 * float(res):.2f} m/s）" if theory is not None else ""), out)
    if not plots:
        return
    plt = _setup_plt()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(v, rows[0], color="#2A6FDB", lw=1.4, label="antenna1（実データ）")
    ax.plot(v, rows[1], color="#D9822B", lw=1.4, label="antenna2（実データ）")
    if theory is not None:
        ax.plot(v, theory, color="#7F7F7F", lw=1.0, ls="--", label=f"ハン窓 {int(N)} パケットの応答（計算）")
    ax.axvspan(-static_v, static_v, color="#BBBBBB", alpha=0.3, label=f"静止として除く |v| < {static_v:g} m/s")
    ax.axhline(-drop_db, color="#333333", lw=0.8, ls=":")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-45, 3)
    ax.set_xlabel("速度 [m/s]")
    ax.set_ylabel("ピークからの強さ [dB]")
    ax.set_title(f"静かな区間（ENMO の小さい {k} 区間）の平均スペクトル: {tag}", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"static_spectrum_{tag}.png"), dpi=110)
    plt.close(fig)


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


def _plot_gradcam(out_dir, tag, X, cams, y, v, window, sp_label, out):
    """Where the CNN looked (Grad-CAM), averaged over the test windows in 3 groups of true ENMO.
    CNN が見た場所（Grad-CAM）。テストの区間を正解の ENMO で 3 つに分けて平均する"""
    try:
        plt = _setup_plt()
        q = np.quantile(y, [1 / 3, 2 / 3])
        groups = [("ENMO 低", y <= q[0]), ("ENMO 中", (y > q[0]) & (y <= q[1])), ("ENMO 高", y > q[1])]
        fig, ax = plt.subplots(3, 3, figsize=(14, 10), gridspec_kw=dict(width_ratios=[1, 1, 0.7]))
        ext = [0, window, v[0], v[-1]]
        for i, (name, m) in enumerate(groups):
            if not m.any():
                continue
            img = X[m].mean(axis=(0, 3))   # time x velocity, both antennas / 時間 x 速度（アンテナ 2 本の平均）
            cam = np.nanmean(cams[m], axis=0)
            ax[i, 0].imshow(img.T, aspect="auto", origin="lower", extent=ext, cmap="viridis")
            ax[i, 0].set_title(f"{name}（{m.sum()} 区間、平均 {y[m].mean():.0f} mG）: 入力のドップラー（log10）",
                               loc="left", fontsize=9)
            ax[i, 1].imshow(cam.T, aspect="auto", origin="lower", extent=ext, cmap="inferno", vmin=0)
            ax[i, 1].set_title("Grad-CAM（明るいほど ENMO を上げる方向に効いた）", loc="left", fontsize=9)
            ax[i, 2].plot(cam.mean(axis=0), v, color="#B5487A")
            ax[i, 2].set_title("速度ごとの効き（時間で平均）", loc="left", fontsize=9)
            for a in ax[i, :2]:
                a.set_ylabel("速度 [m/s]")
            ax[i, 2].set_ylim(v[0], v[-1])
        for a in ax[-1, :2]:
            a.set_xlabel("区間の中の時刻 [s]")
        ax[-1, 2].set_xlabel("Grad-CAM（平均）")
        fig.suptitle(f"CNN が見た場所（{sp_label}）")
        fig.tight_layout()
        path = os.path.join(out_dir, f"cnn_gradcam_{tag}.png")
        fig.savefig(path, dpi=110)
        plt.close(fig)
        log(f"      CNN が見た場所の図: {os.path.basename(path)}", out)
        # one real window per group (not averaged), the one closest to the group's median ENMO
        # 平均しない 1 区間ずつの例（グループの中央値に一番近い区間）
        fig, ax = plt.subplots(3, 2, figsize=(12, 10))
        for i, (name, m) in enumerate(groups):
            if not m.any():
                continue
            idx = np.where(m)[0]
            k = idx[np.argmin(np.abs(y[idx] - np.median(y[idx])))]
            img = X[k].mean(axis=-1)
            ax[i, 0].imshow(img.T, aspect="auto", origin="lower", extent=ext, cmap="viridis")
            ax[i, 0].set_title(f"{name}: 1 区間の実際のドップラー（log10、正解 ENMO {y[k]:.0f} mG）", loc="left", fontsize=9)
            ax[i, 1].imshow(img.T, aspect="auto", origin="lower", extent=ext, cmap="gray")
            ax[i, 1].imshow(cams[k].T, aspect="auto", origin="lower", extent=ext, cmap="inferno", alpha=0.55, vmin=0)
            ax[i, 1].set_title("同じ区間の Grad-CAM（重ねて表示）", loc="left", fontsize=9)
            for a in ax[i]:
                a.set_ylabel("速度 [m/s]")
        for a in ax[-1]:
            a.set_xlabel("区間の中の時刻 [s]")
        fig.suptitle(f"1 区間ずつの例（{sp_label}）")
        fig.tight_layout()
        path = os.path.join(out_dir, f"cnn_gradcam_examples_{tag}.png")
        fig.savefig(path, dpi=110)
        plt.close(fig)
        log(f"      1 区間ずつの例: {os.path.basename(path)}", out)
    except Exception as e:  # the figure is optional / 図は失敗しても続ける
        log(f"      注意: Grad-CAM の図を作れませんでした（{e}）", out)


def _split_tag(sp):
    """File-name tag of a split ('' for the usual leave-one-trial-out). / 分け方のファイル名用の印"""
    if sp == "試行ごと":
        return ""
    t = sp.replace("学習 ", "train_").replace(" → テスト ", "_test_").replace("+", "-")
    return re.sub(r"[^\w.-]", "", t) + "_"


def _plots(out_dir, results):
    plt = _setup_plt()
    colors = {"baseline": "#999999", "linear": "#2A6FDB", "rf": "#3A9D5D", "xgboost": "#D9822B",
              "mlp": "#B5487A", "svr": "#1F9E9E", "gpr": "#8C6D31", "cnn": "#6A4FB3",
              "cnn_shuffle": "#A895D8", "cnn_timemean": "#C9BEEA", "cnn1d": "#4B3590", "cnn3s": "#8E44AD",
              "cnn5s": "#5B2C83"}
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
            ax.set_title(f"{tn}  MAE {pt['MAE']:.1f} mG / MAPE {pt['MAPE']:.0f} % / R² {pt['R2']:.2f}", loc="left", fontsize=10)
            ax.set_ylabel("ENMO [mG]")
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
        ax.set_xlabel("正解 [mG]")
        ax.set_ylabel("予測 [mG]")
        ax.set_title(f"全体: MAE {r['MAE']:.1f} mG / MAPE {r['MAPE']:.0f} % / R² {r['R2']:.2f} / r {r['r']:.2f}", fontsize=10)
        ax.legend(fontsize=8)
        fig.suptitle(f"{r['label']}（正解 {TARGET_JA[r.get('target', 'waist')].split('（')[0]}、特徴量 {r['features']}、"
                     f"ドップラー {r['variant']}、区間 {r['window']:g} 秒）"
                     + ("" if r["split"] == "試行ごと" else f"\n{r['split']}"))
        fig.tight_layout()
        jtag = f"_jitter{r['jitter']:g}dB" if r.get("jitter", 0) > 0 else ""
        fig.savefig(os.path.join(out_dir, f"pred_{_split_tag(r['split'])}{r['variant']}_{r['window']:g}s_{r['model']}_"
                                          f"{r['features'].split()[0].replace('/', '_')}{jtag}.png"), dpi=110)
        plt.close(fig)
    # comparison / 比較
    groups = list(dict.fromkeys((r["variant"], r["window"], r["split"]) for r in results))
    nmax = max(sum(1 for r in results if (r["variant"], r["window"], r["split"]) == g) for g in groups)
    fig, ax = plt.subplots(len(groups), 3, figsize=(max(20, 1.8 * nmax), 4.2 * len(groups)), squeeze=False)
    for vi, (var, wl, sp) in enumerate(groups):
        rs = [r for r in results if r["variant"] == var and r["window"] == wl and r["split"] == sp]
        short = {"baseline": "基準\n(平均値)", "linear": "線形", "rf": "RF", "xgboost": "XGBoost",
                 "mlp": "MLP", "svr": "SVR", "gpr": "ガウス\n過程", "cnn": "CNN", "cnn_shuffle": "CNN\n時間入替",
                 "cnn_timemean": "CNN\n時間平均", "cnn1d": "1次元\nCNN", "cnn3s": "CNN\n前後1秒",
                 "cnn5s": "CNN\n前後2秒"}
        labels = [short[r["model"]] + ((f"\n±{r['jitter']:g}dB" if r.get("jitter", 0) > 0 else "")
                                       if r["model"] in ["baseline"] + CNN_KINDS else f"\n特徴量{r['features']}")
                  for r in rs]
        x = np.arange(len(rs))
        cols = [colors[r["model"]] for r in rs]
        for j, (key, unit) in enumerate((("MAE", " [mG]（小さいほど良い）"),
                                         ("MAPE", f" [%]（小さいほど良い。正解 {MAPE_MIN:g} mG 以上の区間）"),
                                         ("R2", "（1 に近いほど良い）"))):
            vals = [r[key] for r in rs]
            ax[vi, j].bar(x, vals, color=cols)
            for xi, val, r in zip(x, vals, rs):
                txt = f"{val:.2f}" if key == "R2" else f"{val:.1f}"
                if r.get("seed_metrics"):  # spread over the seeds / 乱数の種によるばらつき
                    txt += "\n±" + (f"{r['R2_sd']:.2f}" if key == "R2" else f"{r[key + '_sd']:.1f}")
                ax[vi, j].text(xi, val, txt, ha="center", va="bottom" if val >= 0 else "top", fontsize=8)
            ax[vi, j].set_xticks(x)
            ax[vi, j].set_xticklabels(labels, fontsize=9)
            ax[vi, j].set_ylabel(("R²" if key == "R2" else key) + unit, fontsize=9)
            ax[vi, j].set_title(f"ドップラー: {var}、区間 {wl:g} 秒" + ("" if sp == "試行ごと" else f"、{sp}"),
                                loc="left", fontsize=10)
            if key == "R2":
                ax[vi, j].axhline(0, color="#333333", lw=0.8)
    fig.suptitle(f"モデルの比較（正解: {TARGET_JA[results[0].get('target', 'waist')]}。RF = ランダムフォレスト、"
                 "線形 = 線形回帰 (Ridge)）")
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


def parse_jitter(text):
    """--gain_jitter: dB values, 0 = no jitter (e.g. "0,3,6"). / 強さのゆらぎの大きさ [dB]（0 はゆらさない）"""
    if isinstance(text, (int, float)):
        text = str(text)
    try:
        js = [float(x) for x in str(text).split(",") if x.strip()]
    except ValueError:
        raise RegError(f"--gain_jitter は dB の数字をカンマで区切って指定します（{text}）")
    if not js or any(j < 0 or j > 20 for j in js):
        raise RegError(f"--gain_jitter の値は 0〜20 dB です（{text}）")
    return list(dict.fromkeys(js))


def main(argv=None):
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=os.path.join(HERE, "regression_input"), help="入力フォルダ（既定 regression_input）")
    ap.add_argument("--out", default=None, help="保存先（既定 outputs/regression_<日時>）")
    ap.add_argument("--pairs", default=None,
                    help="IMU と CSI の組み合わせの一覧（加速度計算ソフトの一括処理の batch_pairs_*.csv）。"
                         "ドップラーと正解データの名前が対応しないときに使う（入力フォルダの中にあれば自動で使う）")
    ap.add_argument("--exclude", default=EXCLUDE_DEFAULT,
                    help=f"入力フォルダの中で読まないフォルダの名前（カンマ区切り。既定「{EXCLUDE_DEFAULT}」。空にすると全部読む）")
    ap.add_argument("--doppler", choices=list(VARIANT_DIRS) + ["all"], default="nowinnorm",
                    help="使うドップラーの版（既定 nowinnorm。all で 3 つとも）")
    ap.add_argument("--models", default=",".join(ALL_MODELS),
                    help=f"使うモデル（既定 {','.join(ALL_MODELS)}。ほかに cnn3s（前後 1 秒も入力）、cnn5s（前後 2 秒も入力）など）")
    ap.add_argument("--static_v", type=float, default=STATIC_V,
                    help=f"特徴量 F で静止とみなす速さ [m/s]（|v| がこれ未満のビンを静止として除く。既定 {STATIC_V:g} = "
                         "ハン窓の主ローブの端、速度分解能の約 2 倍）")
    ap.add_argument("--deriv_window", default=f"{DERIV_WINDOW:g}",
                    help=f"特徴量 F の微分の窓 [s]（Savitzky-Golay。既定 {DERIV_WINDOW:g}。0.1,0.2,0.4 のように複数で比べる）")
    ap.add_argument("--features", default="F1,F2a,F2b",
                    help="CNN 以外のモデルの特徴量（既定 F1,F2a,F2b = 特徴量 F: 動いているビンで足したドップラーの時間微分の "
                         "1 秒平均。F1: log10 Σ_動 D、F2a: Σ_動 |v| D ÷ Σ_静止 D、F2b: 平均の速さ。前の特徴量 A: 平均スペクトル、"
                         "B: 物理特徴量、C: 速さの時間微分、D: 1 秒の中の時間変化、E: 強さをそろえたスペクトルも指定すれば使える。"
                         "組み合わせは F1+F2a、A+D など。B+D@1 で前後 1 区間の特徴量も並べる）")
    ap.add_argument("--target", choices=list(TARGETS), default="waist",
                    help="正解にする ENMO（既定 waist = 腰の ENMO。body = 全身の ENMO（腰・大腿・下腿・手首の 4 台の合計。"
                         "加速度計算ソフト v2.0 で 4 台そろった試行だけ）")
    ap.add_argument("--windows", default="1", help="区間の長さ [s]（既定 1 = 1 秒ごとに推定。カンマ区切りで複数も可）")
    ap.add_argument("--split", action="append", default=None,
                    help="学習とテストの分け方「学習:テスト」（何回でも指定できる）。データセット名・試行名・A2_* のような"
                         "書き方をカンマで並べる。例 --split A1:A2 --split A1,A2_XR1:A2_XR2。cv で今までの"
                         "「1 試行をテスト、残りで学習」。指定しなければ cv")
    ap.add_argument("--blocks", type=int, default=5, help="試行が 1 本のとき、時間を分けるかたまりの数（既定 5）")
    ap.add_argument("--epochs", type=int, default=200, help="CNN の最大エポック数（既定 200。早期終了あり）")
    ap.add_argument("--cnn_seeds", type=int, default=1,
                    help="CNN を乱数の種を変えて何回学習するか（既定 1。例 5 で平均とばらつきを出し、予測は 5 回の平均）")
    ap.add_argument("--mape_min", type=float, default=MAPE_MIN,
                    help=f"MAPE（平均絶対パーセント誤差）を計算するときに除く、正解の ENMO の下限 [mG]（既定 {MAPE_MIN:g}。"
                         "静止で ENMO ≈ 0 の区間は誤差 ÷ 正解が発散するため除く）")
    ap.add_argument("--gain_jitter", default="0",
                    help="CNN の学習のとき、区間ごと・アンテナごとに動いている速度ビン（|v| ≥ --static_v）の強さを最大 ±この dB "
                         "ランダムに変える（既定 0 = 変えない。テストの入力はそのまま）。測定した回ごとにドップラーの強さが違っても"
                         "当たるようにするため。0,3,6 のように複数で比べる")
    ap.add_argument("--cnn_study", action="store_true",
                    help="CNN が効く理由を確かめる: 時間の順番を入れ替えた CNN・時間で平均した CNN・1 次元 CNN・"
                         "前後 1 秒も入れた CNN を足し、CNN が見た場所の図（Grad-CAM）を作る")
    ap.add_argument("--seed", type=int, default=0, help="乱数の種（既定 0）")
    ap.add_argument("--importance", action="store_true",
                    help="ランダムフォレストと XGBoost の特徴量の重要度を出す（importance.csv・importance_groups.csv・図）")
    ap.add_argument("--levels_only", action="store_true",
                    help="試行ごとの強さ（静止成分・動いている成分・平均の速さ）の表だけを出して終わる（学習しない。trial_levels.csv）")
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
