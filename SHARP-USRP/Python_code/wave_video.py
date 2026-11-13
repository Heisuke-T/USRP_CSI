# -*- coding: utf-8 -*-
"""Video of one trial's ENMO drawn as time goes on, to put next to the camera video of the measurement.
1 試行の ENMO を時間に合わせて描いていく動画（測定中のカメラの動画と並べて、加速度が動きと合っているか確かめる）

  # 1 s ENMO from a regression result (measured whole-body ENMO, Wi-Fi and waist-accelerometer estimates)
  # 回帰の結果の 1 秒ごとの ENMO（実測の全身 ENMO・Wi-Fi の推定・腰の加速度計からの推定）
  python wave_video.py --trial A5_XR1 --pred shared_results/regression_20261114_014100/predictions.csv

  # + the 50 Hz ENMO of each sensor (acceleration calculator v2.3 *_imu_waveform.csv)
  # ＋ 各センサの 50 Hz の ENMO（加速度計算ソフト v2.3 の *_imu_waveform.csv）
  python wave_video.py --trial A5_XR1 --pred ...predictions.csv --wave regression_input/A5_XR1/A5_XR1_imu_waveform.csv

--offset S: the video's time = CSI time + S (match the camera: if the camera shows the start of the CSI capture
at 3.2 s, use --offset 3.2). --window 10: scroll a 10 s window instead of showing the whole trial.
--offset S: 動画の時刻 = CSI の時刻 + S（カメラの動画で CSI の取得開始が 3.2 秒なら --offset 3.2）。
--window 10: 試行全体ではなく 10 秒の窓をスクロールする
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation, font_manager

FONTS = ["BIZ UDPGothic", "Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP", "IPAexGothic", "DejaVu Sans"]
PARTS = [("waist", "腰"), ("thigh", "大腿"), ("shank", "下腿"), ("wrist", "手首")]
K, BLUE, ORG = "#262626", "#0B4F8A", "#C55A11"
PART_C = {"waist": "#C55A11", "thigh": "#2E7D32", "shank": "#6A1B9A", "wrist": "#0B4F8A"}


def set_font():
    for f in ("/root/.fonts/BIZUDPGothic-Regular.ttf", "/root/.fonts/BIZUDPGothic-Bold.ttf"):
        if os.path.exists(f):
            font_manager.fontManager.addfont(f)
    have = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = [f for f in FONTS if f in have] or ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def load_pred(path, trial):
    """1 s ENMO of the trial: t (window centre), measured, Wi-Fi, waist accelerometer (when present).
    試行の 1 秒ごとの ENMO: 時刻（区間の中央）・実測・Wi-Fi・腰の加速度計（あれば）"""
    p = pd.read_csv(path)
    p = p[p.trial == trial]
    if p.empty:
        sys.exit(f"{path} に試行 {trial} がありません")
    tc = [c for c in p.columns if c.startswith("true_")][0]
    pc = [c for c in p.columns if c.startswith("pred_")][0]
    out = {}
    for model, key in (("cnn3s", "wifi"), ("cnn_imu3s", "waist")):
        g = p[p.model == model].sort_values("start_csi(s)")
        if len(g):
            out["t"] = g["start_csi(s)"].values + 0.5 * (g["end_csi(s)"].values - g["start_csi(s)"].values)
            out["true"] = g[tc].values
            out[key] = g[pc].values
    if "t" not in out:
        sys.exit(f"{path} に {trial} の CNN（前後1秒）の予測がありません")
    return out


def load_wave(path):
    """50 Hz ENMO of each sensor from *_imu_waveform.csv: max(|a| − 1 g, 0) × 1000 [mG].
    各センサの 50 Hz の ENMO"""
    w = pd.read_csv(path)
    t = w["t_csi(s)"].values
    out = {}
    for part, _ in PARTS:
        cols = [f"{part}_Acc{a}(g)" for a in "XYZ"]
        if all(c in w.columns for c in cols):
            a = np.sqrt(sum(w[c].values ** 2 for c in cols))
            out[part] = np.maximum(a - 1.0, 0.0) * 1000.0
    if not out:
        sys.exit(f"{path} に加速度の列（waist_AccX(g) など）がありません")
    return t, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trial", required=True, help="試行名（例 A5_XR1）")
    ap.add_argument("--pred", help="回帰の結果の predictions.csv（1 秒ごとの ENMO）")
    ap.add_argument("--wave", help="加速度計算ソフト v2.3 の <試行>_imu_waveform.csv（50 Hz）")
    ap.add_argument("--out", help="出力の mp4（既定 <試行>_wave.mp4）")
    ap.add_argument("--offset", type=float, default=0.0, help="動画の時刻 = CSI の時刻 + offset [s]（既定 0）")
    ap.add_argument("--window", type=float, default=0.0, help="スクロールする窓 [s]（既定 0 = 試行全体を表示）")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--speed", type=float, default=1.0, help="再生速度（既定 1 = 実時間）")
    o = ap.parse_args()
    if not o.pred and not o.wave:
        sys.exit("--pred か --wave（または両方）を指定してください")
    set_font()
    P = load_pred(o.pred, o.trial) if o.pred else None
    W = load_wave(o.wave) if o.wave else None
    starts = ([P["t"][0] - 0.5] if P else []) + ([W[0][0]] if W else [])
    ends = ([P["t"][-1] + 0.5] if P else []) + ([W[0][-1]] if W else [])
    t0, t1 = min(starts), max(ends)

    n_rows = (1 if P else 0) + (len(W[1]) if W else 0)
    hr = ([2.2] if P else []) + [1.0] * (len(W[1]) if W else 0)
    fig, axs = plt.subplots(n_rows, 1, figsize=(12.8, 7.2), dpi=100, sharex=True, gridspec_kw=dict(height_ratios=hr))
    axs = np.atleast_1d(axs)
    lines, cursors = [], []
    i = 0
    if P:
        ax = axs[0]
        for key, col, ls, lab in (("true", K, "-", "実測（加速度計 4台の合計）"), ("wifi", BLUE, "-", "Wi-Fi（ドップラー）の推定"),
                                  ("waist", ORG, "--", "腰の加速度計からの推定")):
            if key in P:
                ln, = ax.plot([], [], color=col, ls=ls, lw=2.2 if key == "true" else 1.8, label=lab)
                lines.append((ln, P["t"], P[key]))
        top = max(np.nanmax(P[k]) for k in ("true", "wifi", "waist") if k in P)
        ax.set_ylim(0, top * 1.1)
        ax.set_ylabel("全身の\nENMO\n[mG]\n1秒ごと", fontsize=10, rotation=0, ha="right", va="center")
        ax.legend(loc="upper left", fontsize=10, ncol=3, frameon=False)
        i = 1
    if W:
        tw, ew = W
        for part, jp in PARTS:
            if part not in ew:
                continue
            ax = axs[i]
            ln, = ax.plot([], [], color=PART_C[part], lw=1.0)
            lines.append((ln, tw, ew[part]))
            ax.set_ylim(0, max(np.percentile(ew[part], 99.5) * 1.15, 50))
            ax.set_ylabel(f"{jp}\n[mG]", fontsize=10, rotation=0, ha="right", va="center")
            i += 1
        axs[1 if P else 0].set_title("各センサの ENMO（50 Hz）", fontsize=10, loc="left")
    for ax in axs:
        cursors.append(ax.axvline(t0 + o.offset, color="#E53935", lw=1.2))
        ax.tick_params(labelsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[-1].set_xlabel("時刻 [s]" + (f"（CSI の時刻 + {o.offset:g} s）" if o.offset else "（CSI の時刻）"), fontsize=10)
    if not o.window:
        axs[-1].set_xlim(t0 + o.offset, t1 + o.offset)
    clock = fig.text(0.985, 0.975, "", ha="right", va="top", fontsize=20, weight="bold", color="#E53935")
    fig.suptitle(o.trial, x=0.015, y=0.985, ha="left", fontsize=14, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    n = int(np.ceil((t1 - t0) / o.speed * o.fps)) + 1

    def draw(k):
        now = t0 + k * o.speed / o.fps
        for ln, t, y in lines:
            m = t <= now
            tt, yy = t[m], y[m]
            if m.any() and m.sum() < len(t):      # grow smoothly into the next point / 次の点まで滑らかに伸ばす
                j = m.sum()
                yy = np.append(yy, np.interp(now, t[j - 1:j + 1], y[j - 1:j + 1]))
                tt = np.append(tt, now)
            ln.set_data(tt + o.offset, yy)
        for c in cursors:
            c.set_xdata([now + o.offset, now + o.offset])
        if o.window:
            a = max(t0, now - o.window * 0.8) + o.offset
            axs[-1].set_xlim(a, a + o.window)
        clock.set_text(f"t = {now + o.offset:5.1f} s")
        return []

    writer = animation.FFMpegWriter(fps=o.fps, bitrate=3000)
    out = o.out or f"{o.trial}_wave.mp4"
    with writer.saving(fig, out, dpi=100):
        for k in range(n):
            draw(k)
            writer.grab_frame()
            if k % (o.fps * 10) == 0:
                print(f"  {k / n * 100:3.0f} %", flush=True)
    print(f"保存しました: {out}（{n / o.fps:.0f} 秒、{o.fps} fps）")


if __name__ == "__main__":
    main()
