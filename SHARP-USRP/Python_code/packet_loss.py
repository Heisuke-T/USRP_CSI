"""
    受信レートの低下 (パケット欠落) を疑似的に作るマスク

    位相サニタイゼーション済みの CSI (processed_phase/*.mat) のパケットを
    「受信できた / 欠落した」に振り分けるマスクを作り、欠落したパケットを
    取り除く (または埋める)。受信が止まる時間の長さと規則性で、次の 4 通りを
    用意している。どれも欠落率 --loss_rate で強さを変えられる。

      パターン    受信の様子 (|: 受信, .: 欠落)        主な原因
      random     |||.||||.|||||.|||.||               低 SNR、他の端末との衝突
                 1 パケットずつ独立に欠ける (欠落率 p)
      burst      |||||||.........||||||||..........  他の通信のチャネル占有、遮蔽
                 平均 --burst_ms の間まとめて欠け、その後回復する。
                 欠ける区間・受信する区間の長さは指数分布 (Gilbert-Elliott
                 モデルを時間で表したもの)。--burst_dist fixed で欠ける長さを固定
      periodic   ||||||||||||.....||||||||||||.....  端末・AP の定期スキャン
                 --period_s ごとに一定時間まとめて欠ける。
                 欠ける長さ = 欠落率 x 周期 (--burst_ms で直接指定してもよい)
      sustained  |...|...|...|...|...|...|...|...   混雑で送信機が CSMA/CA で待機し続ける
                 途切れはしないが、受信レートが (1 - 欠落率) 倍に下がる。
                 受信するパケットは時間的にほぼ等間隔に選ぶ

    --loss_start_s / --loss_duration_s で低下させる区間を限れる (既定は全区間)。

    欠落したパケットの扱い (--loss_fill):
      drop    取り除く (既定)。実際に受信が欠けたときにパイプラインが受け取る
              データと同じ。time_sec も欠落後のものになるので、--resample nudft
              なら空白の時間も正しく扱われる
      hold    直前に受信したパケットで埋める (パケットの時刻の並びは保つ)
      interp  前後の受信パケットから線形補間で埋める (同上)

    2 台の受信機 (antenna1, antenna2) の扱い (--loss_antennas):
      random は受信側の要因 (SNR、衝突) なので受信機ごとに独立に欠ける。
      burst / periodic / sustained は送信側の要因 (チャネル占有、スキャン、待機)
      で送られなかったパケットなので、同じ測定の 2 台で同じパケットが欠ける。
      既定 (auto) はこの通り。shared / independent で固定もできる。

    使い方:
      # 単体で: processed_phase の .mat に欠落を入れて別のフォルダへ書き出す
      python packet_loss.py ./processed_phase/S2_*.mat --out_dir ./processed_phase_loss/ \\
          --loss burst --loss_rate 0.2 --burst_ms 100 --preview

      # ドップラー計算で直接 (CSI_doppler_computation.py に同じオプションがある)
      python CSI_doppler_computation.py ./processed_phase/ . ./doppler_traces_loss/ \\
          10 10 31 1 -3.0 --despike --loss random --loss_rate 0.3

      # 活動認識まで: 欠落率を振って正解率を比べる (run_har.py)
      python run_har.py --train S1 --test S2,S3 --noise_level -3.0 --despike \\
          --loss burst --loss_rate 0.1,0.2,0.4 --burst_ms 100

    Copyright (C) 2026 Heisuke Takeda
    Released under the GNU GPL v3 (see LICENSE).
"""

import argparse
import glob
import json
import os
import re
import zlib

import numpy as np

PATTERNS = ('random', 'burst', 'periodic', 'sustained')
FILL_MODES = ('drop', 'hold', 'interp')


# ----------------------------------------------------------------------------
# マスクの生成 (True = 受信, False = 欠落)
# ----------------------------------------------------------------------------

def _check_rate(rate, pattern):
    if rate is None:
        raise ValueError(f'{pattern}: --loss_rate を指定してください')
    if not 0.0 <= rate < 1.0:
        raise ValueError(f'--loss_rate は 0 以上 1 未満で指定してください (指定値 {rate})')


def random_loss(n, rate, rng):
    """1 パケットずつ独立に、確率 rate で欠ける。"""
    _check_rate(rate, 'random')
    return rng.random(n) >= rate


def burst_loss(t, rate, burst_ms, rng, dist='exp'):
    """平均 burst_ms の間まとめて欠け、その後回復する (時間で表した Gilbert-Elliott)。

    欠ける区間の平均長 L と欠落率 p から、受信する区間の平均長を
    L (1 - p) / p とすると、長い時間で見た欠落の割合が p になる。
    dist='exp' では両方の長さが指数分布、'fixed' では欠ける長さを L に固定
    (受信する区間は指数分布のまま) する。
    """
    _check_rate(rate, 'burst')
    if burst_ms is None or burst_ms <= 0:
        raise ValueError('burst: --burst_ms (欠ける区間の平均長 [ms]) を正の値で指定してください')
    keep = np.ones(t.size, dtype=bool)
    if rate == 0 or t.size == 0:
        return keep

    bad_mean = burst_ms / 1e3
    good_mean = bad_mean * (1 - rate) / rate
    t0, t1 = float(t.min()), float(t.max())

    def draw(bad):
        if bad and dist == 'fixed':
            return bad_mean
        return rng.exponential(bad_mean if bad else good_mean)

    # 定常状態から始める: 先頭が欠ける区間に入っている確率は p。最初の区間は
    # 途中から始まったものとして残りの長さを使う (指数分布は無記憶なので
    # そのまま、固定長なら一様に短くする)
    bad = rng.random() < rate
    d = draw(bad)
    if bad and dist == 'fixed':
        d *= rng.random()
    edges, states = [], []
    cur = t0
    while True:
        states.append(bad)
        cur += d
        edges.append(cur)
        if cur > t1:
            break
        bad = not bad
        d = draw(bad)
    idx = np.searchsorted(np.asarray(edges), t, side='right')
    idx = np.minimum(idx, len(states) - 1)
    keep[np.asarray(states)[idx]] = False
    return keep


def periodic_loss(t, period_s, rng, rate=None, burst_ms=None):
    """period_s ごとに一定時間 (= rate x 周期, または burst_ms) まとめて欠ける。

    最初の欠落がどこから始まるか (位相) は乱数で決める。
    """
    if period_s is None or period_s <= 0:
        raise ValueError('periodic: --period_s (欠落の周期 [s]) を正の値で指定してください')
    dur = periodic_burst_s(period_s, rate, burst_ms)
    offset = rng.uniform(0, period_s)
    return ((t - t.min() + offset) % period_s) >= dur


def periodic_burst_s(period_s, rate=None, burst_ms=None):
    """periodic の 1 回あたりの欠落時間 [s]。"""
    if (rate is None) == (burst_ms is None):
        raise ValueError('periodic: --loss_rate と --burst_ms のどちらか一方を指定してください')
    if burst_ms is not None:
        dur = burst_ms / 1e3
        if not 0 <= dur < period_s:
            raise ValueError(f'periodic: --burst_ms ({burst_ms}) は周期より短くしてください')
        return dur
    _check_rate(rate, 'periodic')
    return rate * period_s


def sustained_loss(t, rate, rng):
    """途切れはしないが受信レートが (1 - rate) 倍に下がる。

    目標のレートで等間隔に並べた時刻を作り、それぞれの時刻以降で最初に
    届いたパケットを受信したものとする (CSMA/CA で待たされて送信間隔が
    延びた状態)。同じパケットが選ばれた分は 1 つにまとめる。
    """
    _check_rate(rate, 'sustained')
    keep = np.zeros(t.size, dtype=bool)
    if t.size == 0 or rate == 0:
        keep[:] = True
        return keep
    order = np.argsort(t, kind='stable')
    ts = t[order]
    span = ts[-1] - ts[0]
    if span <= 0:
        keep[:] = True
        return keep
    # 1 パケットあたりの平均間隔から、低下後の間隔を決める
    interval = span / (ts.size - 1) / (1 - rate)
    grid = ts[0] + rng.uniform(0, interval) + np.arange(0, span, interval)
    sel = np.unique(np.searchsorted(ts, grid, side='left'))
    sel = sel[sel < ts.size]
    keep[order[sel]] = True
    return keep


def make_mask(t, pattern, rate=None, burst_ms=None, period_s=None,
              burst_dist='exp', start_s=None, duration_s=None, rng=None):
    """パケット時刻 t [s] から受信マスク (True = 受信) を作る。

    start_s / duration_s を与えると、先頭のパケットからの経過時間がその区間に
    入るパケットだけを低下させる (区間外はすべて受信)。
    """
    t = np.asarray(t, dtype=float).ravel()
    rng = rng if rng is not None else np.random.default_rng(0)
    if pattern not in PATTERNS:
        raise ValueError(f'欠落パターンは {", ".join(PATTERNS)} のどれかです ({pattern})')

    active = np.ones(t.size, dtype=bool)
    if t.size:
        rel = t - t.min()
        if start_s is not None:
            active &= rel >= start_s
        if duration_s is not None:
            active &= rel < (start_s or 0.0) + duration_s

    ta = t[active]
    if pattern == 'random':
        k = random_loss(ta.size, rate, rng)
    elif pattern == 'burst':
        k = burst_loss(ta, rate, burst_ms, rng, dist=burst_dist)
    elif pattern == 'periodic':
        k = periodic_loss(ta, period_s, rng, rate=rate, burst_ms=burst_ms)
    else:
        k = sustained_loss(ta, rate, rng)

    keep = np.ones(t.size, dtype=bool)
    keep[active] = k
    return keep


# ----------------------------------------------------------------------------
# マスクの適用
# ----------------------------------------------------------------------------

def fill_lost(x, keep, mode):
    """欠落したパケット (keep=False の行) を埋める。x: [パケット x ...] の複素数。

    hold:   直前に受信したパケットで埋める (先頭が欠けていれば最初の受信で)
    interp: 前後の受信パケットから線形補間 (実部・虚部を別々に)
    """
    if keep.all():
        return x
    good = np.flatnonzero(keep)
    if good.size == 0:
        raise ValueError('すべてのパケットが欠落しました。欠落率を下げてください')
    idx = np.arange(x.shape[0])
    if mode == 'hold':
        src = good[np.maximum(np.searchsorted(good, idx, side='right') - 1, 0)]
        return x[src]
    if mode == 'interp':
        flat = x.reshape(x.shape[0], -1)
        out = np.empty_like(flat)
        for k in range(flat.shape[1]):
            out[:, k] = (np.interp(idx, good, flat[good, k].real)
                         + 1j * np.interp(idx, good, flat[good, k].imag))
        return out.reshape(x.shape)
    raise ValueError(f'埋め方は hold / interp です ({mode})')


def apply_mask(x, time_sec, keep, fill='drop'):
    """マスクを複素 CFR [パケット x サブキャリア] と時刻に適用する。

    戻り値: (適用後の CFR, 適用後の時刻)。drop 以外はパケット数を変えない。
    """
    if fill not in FILL_MODES:
        raise ValueError(f'--loss_fill は {", ".join(FILL_MODES)} のどれかです ({fill})')
    if not keep.any():
        raise ValueError('すべてのパケットが欠落しました。欠落率を下げてください')
    if fill == 'drop':
        return x[keep], (None if time_sec is None else time_sec[keep])
    return fill_lost(x, keep, fill), time_sec


def loss_stats(t, keep):
    """欠落の実績 (実際の欠落率、連続して欠けた長さ、受信レートなど)。"""
    t = np.asarray(t, dtype=float).ravel()
    n = int(keep.size)
    lost = ~keep
    # 連続して欠けた区間 (ラン) の長さ
    d = np.diff(np.concatenate(([0], lost.astype(np.int8), [0])))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    run_pkts = ends - starts
    span = float(t.max() - t.min()) if t.size > 1 else 0.0
    st = dict(
        n_packets=n,
        n_received=int(keep.sum()),
        n_lost=int(lost.sum()),
        loss_rate=float(lost.mean()) if n else 0.0,
        n_bursts=int(run_pkts.size),
        mean_burst_packets=float(run_pkts.mean()) if run_pkts.size else 0.0,
        max_burst_packets=int(run_pkts.max()) if run_pkts.size else 0,
        rate_before_pps=(n - 1) / span if span > 0 else None,
        rate_after_pps=(int(keep.sum()) - 1) / span if span > 0 else None,
    )
    tk = np.sort(t[keep])
    if tk.size > 1:
        gaps = np.diff(tk)
        st['max_gap_ms'] = float(gaps.max() * 1e3)
        st['median_gap_ms'] = float(np.median(gaps) * 1e3)
    if run_pkts.size and t.size > 1:
        # ラン 1 つが占める時間 = 前後の受信パケットの間隔
        ts = np.sort(t)
        lo = np.maximum(starts - 1, 0)
        hi = np.minimum(ends, n - 1)
        dur = ts[hi] - ts[lo]
        st['mean_burst_ms'] = float(dur.mean() * 1e3)
        st['max_burst_ms'] = float(dur.max() * 1e3)
    return st


# ----------------------------------------------------------------------------
# コマンドラインの共通部分
# ----------------------------------------------------------------------------

def add_loss_args(parser, sweep=False):
    """欠落のオプションを parser に追加する (packet_loss.py, CSI_doppler_computation.py,
    run_har.py で共通)。sweep=True なら --loss_rate にカンマ区切りで複数指定できる。"""
    g = parser.add_argument_group('受信レートの低下 (パケット欠落) の模擬')
    g.add_argument('--loss', choices=PATTERNS, default=None,
                   help='欠落のパターン。random: 1 パケットずつ散発的 / burst: まとめて途切れる / '
                        'periodic: 周期的に途切れる / sustained: 途切れないがレートが下がる')
    if sweep:
        g.add_argument('--loss_rate', default=None,
                       help='欠落率 (0〜1)。カンマ区切りで複数指定すると、それぞれで評価して比べる '
                            '(例: 0.1,0.2,0.4)')
    else:
        g.add_argument('--loss_rate', type=float, default=None, help='欠落率 (0〜1)')
    g.add_argument('--burst_ms', type=float, default=None,
                   help='burst: 欠ける区間の平均長 [ms] / periodic: 1 回の欠落時間 [ms] '
                        '(--loss_rate の代わりに指定できる)')
    g.add_argument('--burst_dist', choices=['exp', 'fixed'], default='exp',
                   help='burst の欠ける長さの分布。exp: 指数分布 (既定) / fixed: 常に --burst_ms')
    g.add_argument('--period_s', type=float, default=None,
                   help='periodic: 欠落の周期 [s] (例: 30)')
    g.add_argument('--loss_start_s', type=float, default=None,
                   help='低下させる区間の始まり (先頭のパケットからの秒数)。既定は先頭から')
    g.add_argument('--loss_duration_s', type=float, default=None,
                   help='低下させる区間の長さ [s]。既定は最後まで')
    g.add_argument('--loss_fill', choices=FILL_MODES, default='drop',
                   help='欠落したパケットの扱い。drop: 取り除く (既定、実際の受信と同じ) / '
                        'hold: 直前の受信で埋める / interp: 前後から線形補間')
    g.add_argument('--loss_antennas', choices=['auto', 'shared', 'independent'], default='auto',
                   help='同じ測定の antenna1/antenna2 で同じパケットを欠かすか。auto (既定): '
                        'random は独立、それ以外 (送信側の要因) は共通')
    g.add_argument('--loss_seed', type=int, default=0, help='乱数の種 (既定 0)')
    return g


def check_loss_args(args, rate=None):
    """パターンごとに必要なオプションが揃っているかを確かめる (足りなければ ValueError)。"""
    rate = args.loss_rate if rate is None else rate
    if args.loss is None:
        return
    if args.loss == 'periodic':
        if args.period_s is None or args.period_s <= 0:
            raise ValueError('periodic には --period_s (欠落の周期 [s]) が必要です')
        periodic_burst_s(args.period_s, rate, args.burst_ms)
        return
    _check_rate(rate, args.loss)
    if args.loss == 'burst' and (args.burst_ms is None or args.burst_ms <= 0):
        raise ValueError('burst には --burst_ms (欠ける区間の平均長 [ms]) が必要です')


def file_seed(name, args):
    """ファイルごとの乱数の種。

    測定ごとに違う欠け方にしつつ、再実行では同じになるようにファイル名から作る。
    2 台の受信機で欠落を共通にする場合は、アンテナ名を除いた名前から作るので
    antenna1 と antenna2 で同じ乱数列 (= 同じ時刻の欠落) になる。
    """
    shared = (args.loss_antennas == 'shared'
              or (args.loss_antennas == 'auto' and args.loss != 'random'))
    key = os.path.basename(name)
    if shared:
        key = re.sub(r'antenna\d+', 'antenna', key)
    return (zlib.crc32(key.encode('utf-8')) + args.loss_seed * 1000003) % (2 ** 32)


def mask_for(name, time_sec, args, n=None, Tc=None):
    """args の設定でファイル name の受信マスクを作る。

    time_sec が無い場合は、random なら時刻は不要、それ以外は Tc から等間隔の
    時刻を作る。
    """
    if time_sec is None:
        if args.loss != 'random' and Tc is None:
            raise ValueError(f'{name}: time_sec が無いため {args.loss} の欠落を作れません '
                             '(--Tc で間隔を指定してください)')
        time_sec = np.arange(n) * (Tc or 1.0)
    rng = np.random.default_rng(file_seed(name, args))
    keep = make_mask(time_sec, args.loss, rate=args.loss_rate, burst_ms=args.burst_ms,
                     period_s=args.period_s, burst_dist=args.burst_dist,
                     start_s=args.loss_start_s, duration_s=args.loss_duration_s, rng=rng)
    return keep, time_sec


def loss_cli_args(args, rate):
    """run_har.py から CSI_doppler_computation.py に渡す引数列。"""
    out = ['--loss', args.loss]
    if rate is not None:
        out += ['--loss_rate', rate]
    for opt in ('burst_ms', 'period_s', 'loss_start_s', 'loss_duration_s'):
        v = getattr(args, opt)
        if v is not None:
            out += [f'--{opt}', v]
    out += ['--burst_dist', args.burst_dist, '--loss_fill', args.loss_fill,
            '--loss_antennas', args.loss_antennas, '--loss_seed', args.loss_seed]
    return [str(v) for v in out]


def describe(args, rate=None):
    """結果の表に載せる短い説明 (例: burst 欠落率 0.2, 平均 100 ms)。"""
    rate = args.loss_rate if rate is None else rate
    if args.loss is None:
        return 'なし'
    s = args.loss
    if rate is not None:
        s += f' 欠落率 {float(rate):g}'
    if args.loss == 'burst':
        s += f', {"" if args.burst_dist == "fixed" else "平均 "}{args.burst_ms:g} ms'
    if args.loss == 'periodic':
        dur = periodic_burst_s(args.period_s, None if rate is None else float(rate), args.burst_ms)
        s += f', {args.period_s:g} s ごとに {dur * 1e3:g} ms'
    if args.loss_start_s is not None or args.loss_duration_s is not None:
        s0 = args.loss_start_s or 0.0
        s += f', {s0:g} s〜' + (f'{s0 + args.loss_duration_s:g} s' if args.loss_duration_s else '')
    if args.loss_fill != 'drop':
        s += f', {args.loss_fill} で補完'
    return s


def tag(args, rate=None):
    """欠落の設定を表すフォルダ名 (例: burst_r0.2_b100ms)。"""
    rate = args.loss_rate if rate is None else rate
    parts = [args.loss]
    if rate is not None:
        parts.append(f'r{float(rate):g}')
    if args.burst_ms is not None:
        parts.append(f'b{args.burst_ms:g}ms')
    if args.loss == 'burst' and args.burst_dist == 'fixed':
        parts.append('fixed')
    if args.period_s is not None:
        parts.append(f'T{args.period_s:g}s')
    if args.loss_start_s is not None:
        parts.append(f'from{args.loss_start_s:g}s')
    if args.loss_duration_s is not None:
        parts.append(f'dur{args.loss_duration_s:g}s')
    if args.loss_fill != 'drop':
        parts.append(args.loss_fill)
    if args.loss_antennas != 'auto':
        parts.append(f'ant-{args.loss_antennas}')
    if args.loss_seed:
        parts.append(f'seed{args.loss_seed}')
    return '_'.join(parts)


# ----------------------------------------------------------------------------
# 図 (受信の様子)
# ----------------------------------------------------------------------------

def plot_mask(t, keep, out_png, title='', zoom_s=2.0):
    """受信の様子を描く。上: 先頭 zoom_s 秒の各パケット (青: 受信, 灰: 欠落)、
    下: 全区間の受信レート (100 ms ごと)。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t = np.asarray(t, dtype=float).ravel()
    rel = t - t.min()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4.5),
                                   gridspec_kw=dict(height_ratios=[1, 1.2]))
    z = rel <= zoom_s
    ax1.vlines(rel[z & keep], 0, 1, color='#1f77d0', lw=1.2, label='received')
    ax1.vlines(rel[z & ~keep], 0, 0.25, color='#bbbbbb', lw=1.2, label='lost')
    ax1.set_xlim(0, zoom_s)
    ax1.set_ylim(0, 1.45)          # 上に凡例の余白
    ax1.set_yticks([])
    ax1.set_xlabel('time [s]')
    ax1.legend(loc='upper right', ncol=2, fontsize=8, frameon=False)
    ax1.set_title(title, fontsize=9)

    bin_s = 0.1
    edges = np.arange(0, rel.max() + bin_s, bin_s)
    if edges.size < 2:
        edges = np.array([0, bin_s])
    before, _ = np.histogram(rel, edges)
    after, _ = np.histogram(rel[keep], edges)
    ax2.step(edges[:-1], before / bin_s, where='post', color='#bbbbbb', lw=1, label='original')
    ax2.step(edges[:-1], after / bin_s, where='post', color='#1f77d0', lw=1, label='after loss')
    ax2.set_xlabel('time [s]')
    ax2.set_ylabel('packets / s')
    ax2.set_xlim(0, rel.max() if rel.max() > 0 else 1)
    ax2.set_ylim(0, max(before.max(), 1) / bin_s * 1.35)   # 上に凡例の余白
    ax2.legend(loc='upper right', ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 単体での実行: processed_phase の .mat に欠落を入れて書き出す
# ----------------------------------------------------------------------------

def process_mat(in_path, out_path, args, preview=False):
    import scipy.io as sio

    mdic = sio.loadmat(in_path)
    m = mdic['csi_matrix_processed']           # [パケット x サブキャリア x (振幅, 位相)]
    n = m.shape[0]
    t = None
    if 'time_sec' in mdic:
        t = np.asarray(mdic['time_sec'], dtype=float).ravel()[:n]
    name = os.path.splitext(os.path.basename(in_path))[0]
    keep, t_used = mask_for(name, t, args, n=n, Tc=args.Tc)

    x = m[:, :, 0] * np.exp(1j * m[:, :, 1])
    x, t_out = apply_mask(x, t, keep, args.loss_fill)
    out = np.stack([np.abs(x), np.angle(x)], axis=-1) if args.loss_fill != 'drop' else m[keep]

    st = loss_stats(t_used, keep)
    st.update(source=os.path.basename(in_path), setting=describe(args), fill=args.loss_fill)
    od = {k: v for k, v in mdic.items() if not k.startswith('__')}
    od['csi_matrix_processed'] = out
    if t_out is not None:
        od['time_sec'] = t_out
    od['loss_mask'] = keep                     # 元のパケットごとの受信 (1) / 欠落 (0)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    sio.savemat(out_path, od)
    with open(out_path[:-4] + '_loss.json', 'w', encoding='utf-8') as fp:
        json.dump(st, fp, ensure_ascii=False, indent=2)
    if preview:
        plot_mask(t_used, keep, out_path[:-4] + '_loss.png',
                  title=f'{name}: {args.loss}, loss {st["loss_rate"]:.3f}')
    return st


def print_stats(st, indent='  '):
    s = (f'{indent}欠落 {st["n_lost"]}/{st["n_packets"]} パケット '
         f'(欠落率 {st["loss_rate"]:.3f})')
    if st.get('rate_before_pps'):
        s += f', 受信レート {st["rate_before_pps"]:.0f} -> {st["rate_after_pps"]:.0f} pkt/s'
    print(s)
    if st['n_bursts']:
        s = (f'{indent}連続した欠落 {st["n_bursts"]} 回, '
             f'平均 {st["mean_burst_packets"]:.1f} / 最大 {st["max_burst_packets"]} パケット')
        if 'mean_burst_ms' in st:
            s += f' (平均 {st["mean_burst_ms"]:.1f} / 最大 {st["max_burst_ms"]:.1f} ms)'
        print(s)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('inputs', nargs='+',
                        help='processed_phase の .mat (ワイルドカード可)')
    parser.add_argument('--out_dir', default='./processed_phase_loss/',
                        help='書き出し先 (既定 ./processed_phase_loss/)。同じ名前で書く')
    parser.add_argument('--Tc', type=float, default=None,
                        help='time_sec が無い .mat のパケット間隔 [s]')
    parser.add_argument('--preview', action='store_true',
                        help='受信の様子の図 (<名前>_loss.png) も出す')
    add_loss_args(parser)
    args = parser.parse_args()
    if args.loss is None:
        parser.error('--loss でパターンを指定してください')
    try:
        check_loss_args(args)
    except ValueError as e:
        parser.error(str(e))

    files = []
    for pat in args.inputs:
        files.extend(sorted(glob.glob(pat)) or ([pat] if os.path.exists(pat) else []))
    files = [f for f in dict.fromkeys(files) if f.endswith('.mat')]
    if not files:
        parser.error('入力の .mat が見つかりません')

    print(f'欠落: {describe(args)}  -> {args.out_dir}')
    for f in files:
        out = os.path.join(args.out_dir, os.path.basename(f))
        print(os.path.basename(f))
        st = process_mat(f, out, args, preview=args.preview)
        print_stats(st)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
