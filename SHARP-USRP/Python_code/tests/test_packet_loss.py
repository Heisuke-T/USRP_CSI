"""
    受信レートの低下の模擬 (packet_loss.py) の自己診断テスト。

      1. 4 つのパターンが指定した欠落率どおりに欠けること、欠け方
         (まとめて欠ける長さ、周期、等間隔の受信) が設定どおりであること
      2. 低下させる区間の指定、乱数の再現性、2 台の受信機で欠落を共通に
         するかどうかの切り替え
      3. 欠落したパケットの扱い (drop / hold / interp)
      4. 単体での実行 (processed_phase の .mat に欠落を入れて書き出す)
      5. CSI_doppler_computation.py の --loss で、欠落の実績が _meta.json に残ること
      6. run_har.py --loss で、欠落なしと欠落率ごとの正解率が並ぶこと
         (TensorFlow がある場合のみ)

    実行:  python3 tests/test_packet_loss.py

    Copyright (C) 2026 Heisuke Takeda
    Released under the GNU GPL v3 (see LICENSE).
"""

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
sys.path.insert(0, _CODE)

import packet_loss as pl  # noqa: E402

TC = 0.0058                       # 約 170 pkt/s (実測と同程度)


class TestFailure(AssertionError):
    pass


def check(cond, msg):
    if not cond:
        raise TestFailure(msg)
    print(f'  OK: {msg}')


def times(n=15000, seed=1):
    """実測に似た揺らぎのあるパケット時刻 (変動係数 0.15)。"""
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.gamma(44, TC / 44, size=n))


def loss_args(**kw):
    p = argparse.ArgumentParser()
    pl.add_loss_args(p)
    a = p.parse_args([])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def test_patterns():
    print('\n--- 欠落のパターン ---')
    t = times()
    rng = np.random.default_rng

    k = pl.make_mask(t, 'random', rate=0.3, rng=rng(0))
    st = pl.loss_stats(t, k)
    check(abs(st['loss_rate'] - 0.3) < 0.02, f'random: 欠落率 0.3 ({st["loss_rate"]:.3f})')
    check(st['mean_burst_packets'] < 1.6, f'random: 連続した欠落は短い (平均 {st["mean_burst_packets"]:.2f} パケット)')

    # burst: 長い区間で平均を見る (区間が長いので測定 1 本ぶんでは揺らぐ)
    tl = times(150000)
    k = pl.make_mask(tl, 'burst', rate=0.2, burst_ms=100, rng=rng(0))
    st = pl.loss_stats(tl, k)
    check(abs(st['loss_rate'] - 0.2) < 0.03, f'burst: 欠落率 0.2 ({st["loss_rate"]:.3f})')
    check(80 < st['mean_burst_ms'] < 140,
          f'burst: 途切れの平均 約 100 ms ({st["mean_burst_ms"]:.0f} ms)')
    k = pl.make_mask(tl, 'burst', rate=0.2, burst_ms=100, burst_dist='fixed', rng=rng(0))
    st = pl.loss_stats(tl, k)
    check(abs(st['loss_rate'] - 0.2) < 0.03, f'burst (固定長): 欠落率 0.2 ({st["loss_rate"]:.3f})')
    check(95 < np.median(_run_ms(tl, k)) < 115,
          f'burst (固定長): 途切れの中央値は 100 ms ({np.median(_run_ms(tl, k)):.0f} ms)')

    k = pl.make_mask(t, 'periodic', period_s=10, burst_ms=300, rng=rng(0))
    runs = _run_ms(t, k)
    check(len(runs) in (8, 9, 10), f'periodic: 87 s で 10 s ごとに途切れる ({len(runs)} 回)')
    check(np.all(np.abs(runs - 300) < 20), f'periodic: 1 回 300 ms ({np.round(runs).astype(int)})')
    k2 = pl.make_mask(t, 'periodic', period_s=10, rate=0.03, rng=rng(0))
    check(np.array_equal(k, k2), 'periodic: 欠落率 0.03 x 周期 10 s = 300 ms と同じ')

    k = pl.make_mask(t, 'sustained', rate=0.6, rng=rng(0))
    st = pl.loss_stats(t, k)
    check(abs(st['loss_rate'] - 0.6) < 0.02, f'sustained: 欠落率 0.6 ({st["loss_rate"]:.3f})')
    gaps = np.diff(t[k])
    check(gaps.max() < 4 * TC / 0.4, f'sustained: 途切れない (最大間隔 {gaps.max()*1e3:.1f} ms)')
    check(abs(st['rate_after_pps'] / st['rate_before_pps'] - 0.4) < 0.02,
          f'sustained: 受信レートが 0.4 倍 ({st["rate_before_pps"]:.0f} -> {st["rate_after_pps"]:.0f} pkt/s)')

    for pat in ['random', 'burst', 'sustained']:
        k = pl.make_mask(t, pat, rate=0.0, burst_ms=100, rng=rng(0))
        check(k.all(), f'{pat}: 欠落率 0 なら何も欠けない')

    k = pl.make_mask(t, 'random', rate=0.5, start_s=20, duration_s=30, rng=rng(0))
    rel = t - t[0]
    inside = (rel >= 20) & (rel < 50)
    check(k[~inside].all(), '区間の指定: 区間の外は欠けない')
    check(abs((~k[inside]).mean() - 0.5) < 0.03, '区間の指定: 区間の中は欠落率 0.5')

    for bad in [dict(rate=1.0), dict(rate=-0.1)]:
        try:
            pl.make_mask(t, 'random', rng=rng(0), **bad)
            raise TestFailure(f'{bad} はエラーになるべき')
        except ValueError:
            pass
    check(True, '欠落率が 0〜1 の外ならエラー')


def _run_ms(t, keep):
    """連続した欠落ごとの途切れの長さ [ms] (前後の受信パケットの間隔)。"""
    d = np.diff(np.concatenate(([0], (~keep).astype(int), [0])))
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    s, e = s[(s > 0) & (e < t.size)], e[(s > 0) & (e < t.size)]
    return (t[e] - t[s - 1]) * 1e3


def test_seed_and_antennas():
    print('\n--- 乱数と 2 台の受信機 ---')
    t = times(5000)
    a = loss_args(loss='burst', loss_rate=0.2, burst_ms=100)
    k1, _ = pl.mask_for('S2_W_antenna1_stream_0', t, a)
    k1b, _ = pl.mask_for('S2_W_antenna1_stream_0', t, a)
    k2, _ = pl.mask_for('S2_W_antenna2_stream_0', t, a)
    k3, _ = pl.mask_for('S3_W_antenna1_stream_0', t, a)
    check(np.array_equal(k1, k1b), '同じ設定・同じファイルなら同じ欠け方 (再実行で再現する)')
    check(np.array_equal(k1, k2), 'burst: antenna1 と antenna2 で同じパケットが欠ける (送信側の要因)')
    check(not np.array_equal(k1, k3), '測定が違えば欠け方も違う')
    a.loss_seed = 1
    check(not np.array_equal(k1, pl.mask_for('S2_W_antenna1_stream_0', t, a)[0]),
          '--loss_seed を変えると欠け方が変わる')

    a = loss_args(loss='random', loss_rate=0.2)
    r1, _ = pl.mask_for('S2_W_antenna1_stream_0', t, a)
    r2, _ = pl.mask_for('S2_W_antenna2_stream_0', t, a)
    check(not np.array_equal(r1, r2), 'random: 受信機ごとに独立に欠ける (受信側の要因)')
    a.loss_antennas = 'shared'
    check(np.array_equal(pl.mask_for('S2_W_antenna1_stream_0', t, a)[0],
                         pl.mask_for('S2_W_antenna2_stream_0', t, a)[0]),
          '--loss_antennas shared なら random でも共通')


def test_fill():
    print('\n--- 欠落したパケットの扱い ---')
    n = 10
    x = (np.arange(n) + 1j * np.arange(n))[:, None] * np.ones((1, 3))
    t = np.arange(n) * TC
    keep = np.ones(n, dtype=bool)
    keep[[0, 4, 5]] = False
    y, ty = pl.apply_mask(x, t, keep, 'drop')
    check(y.shape[0] == 7 and np.array_equal(ty, t[keep]), 'drop: 欠落したパケットと時刻を取り除く')
    y, ty = pl.apply_mask(x, t, keep, 'hold')
    check(y.shape == x.shape and y[4, 0] == x[3, 0] and y[5, 0] == x[3, 0] and y[0, 0] == x[1, 0],
          'hold: 直前の受信で埋める (先頭は最初の受信)')
    y, _ = pl.apply_mask(x, t, keep, 'interp')
    check(np.allclose(y[1:], x[1:]), 'interp: 前後から線形補間 (等差の列なら元どおり)')
    try:
        pl.apply_mask(x, t, np.zeros(n, dtype=bool), 'drop')
        raise TestFailure('全部欠けたらエラーになるべき')
    except ValueError:
        check(True, 'すべて欠けたらエラー')


# ----------------------------------------------------------------------------
# 合成の位相サニタイゼーション済み CSI
# ----------------------------------------------------------------------------

DOPPLER_HZ = {'E': [], 'W': [-25.0, 25.0], 'R': [-60.0, 60.0]}


def write_processed(path, act, t, seed):
    """活動ごとに違うドップラー周波数の動き成分を持つ
    csi_matrix_processed [パケット x 57 x (振幅, 位相)] と time_sec。"""
    import scipy.io as sio
    rng = np.random.default_rng(seed)
    n = t.size
    k = np.arange(57)
    h = np.ones((n, 57), dtype=complex) * np.exp(1j * 0.05 * k)[None, :]
    for fd in DOPPLER_HZ[act]:
        ph = rng.uniform(0, 2 * np.pi)
        h += 0.5 * np.exp(1j * (2 * np.pi * fd * t[:, None] + ph + 0.1 * k[None, :]))
    h += 0.05 * (rng.standard_normal(h.shape) + 1j * rng.standard_normal(h.shape))
    m = np.stack([np.abs(h), np.angle(h)], axis=-1)
    sio.savemat(path, {'csi_matrix_processed': m, 'time_sec': t})


def make_processed(root, n=3000):
    d = os.path.join(root, 'processed_phase')
    os.makedirs(d, exist_ok=True)
    seed = 0
    for sess in ['S1', 'S2', 'S3']:
        for act in DOPPLER_HZ:
            # 2 台の受信機は同じパケットを受けている (alignCSI.m で揃えた後と同じ)
            seed += 1
            t = times(n, seed)
            for ant in ['antenna1', 'antenna2']:
                seed += 1
                write_processed(os.path.join(d, f'{sess}_{act}_{ant}_stream_0.mat'), act, t, seed)
    return d


def run(script, *args, cwd):
    cmd = [sys.executable, os.path.join(_CODE, script), *map(str, args)]
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-4000:])
        print(r.stderr[-4000:])
        raise TestFailure(f'{script} が失敗しました')
    return r.stdout


def test_cli(root, proc):
    print('\n--- 単体での実行とドップラー計算 ---')
    import scipy.io as sio
    out_dir = os.path.join(root, 'processed_phase_loss')
    out = run('packet_loss.py', os.path.join(proc, 'S2_W_*.mat'), '--out_dir', out_dir,
              '--loss', 'burst', '--loss_rate', '0.3', '--burst_ms', '80', '--preview', cwd=root)
    check('欠落' in out and '受信レート' in out, '欠落の実績を表示する')
    a1 = sio.loadmat(os.path.join(out_dir, 'S2_W_antenna1_stream_0.mat'))
    a2 = sio.loadmat(os.path.join(out_dir, 'S2_W_antenna2_stream_0.mat'))
    src = sio.loadmat(os.path.join(proc, 'S2_W_antenna1_stream_0.mat'))
    mask = a1['loss_mask'].ravel().astype(bool)
    check(a1['csi_matrix_processed'].shape[0] == mask.sum() == a1['time_sec'].size,
          'drop: パケットと time_sec が受信した分だけになる')
    check(np.array_equal(a1['csi_matrix_processed'], src['csi_matrix_processed'][mask]),
          '受信したパケットの中身は元のまま')
    check(np.array_equal(mask, a2['loss_mask'].ravel().astype(bool)),
          'burst: antenna1 と antenna2 で同じパケットが欠ける')
    with open(os.path.join(out_dir, 'S2_W_antenna1_stream_0_loss.json'), encoding='utf-8') as fp:
        st = json.load(fp)
    check(st['n_lost'] == (~mask).sum(), '_loss.json に欠落の実績が残る')
    check(os.path.exists(os.path.join(out_dir, 'S2_W_antenna1_stream_0_loss.png')),
          '--preview で受信の様子の図を出す')

    run('packet_loss.py', os.path.join(proc, 'S2_W_antenna1*.mat'), '--out_dir', out_dir + '_hold',
        '--loss', 'random', '--loss_rate', '0.3', '--loss_fill', 'hold', cwd=root)
    h = sio.loadmat(os.path.join(out_dir + '_hold', 'S2_W_antenna1_stream_0.mat'))
    check(h['csi_matrix_processed'].shape == src['csi_matrix_processed'].shape,
          'hold: パケット数は変わらない')

    dop = os.path.join(root, 'doppler_loss')
    out = run('CSI_doppler_computation.py', proc + '/', '.', dop, 10, 10, 31, 1, -3.0,
              '--only', 'S2_W_', '--loss', 'sustained', '--loss_rate', '0.5', cwd=root)
    check('欠落の模擬' in out, 'ドップラー計算で欠落を入れられる')
    names = sorted(f for f in os.listdir(dop) if f.endswith('.txt'))
    check(names == ['S2_W_antenna1_stream_0.txt', 'S2_W_antenna2_stream_0.txt'],
          f'--only で対象のファイルだけ計算する ({names})')
    with open(os.path.join(dop, 'S2_W_antenna1_stream_0_meta.json'), encoding='utf-8') as fp:
        meta = json.load(fp)
    pl_meta = meta['packet_loss']
    check(abs(pl_meta['loss_rate'] - 0.5) < 0.02, f'_meta.json に欠落率が残る ({pl_meta["loss_rate"]:.3f})')
    check(meta['Tc'] > 1.8 * TC, f'受信レートが半分になると Tc が約 2 倍 ({meta["Tc"]*1e3:.1f} ms)')

    r = subprocess.run([sys.executable, os.path.join(_CODE, 'CSI_doppler_computation.py'),
                        proc + '/', '.', dop, '10', '10', '31', '1', '-3.0', '--loss', 'burst',
                        '--loss_rate', '0.2'], cwd=root, capture_output=True, text=True)
    check(r.returncode != 0 and '--burst_ms' in r.stderr, 'burst に --burst_ms が無ければ止まる')


def test_run_har(root, proc):
    print('\n--- run_har.py --loss ---')
    work = os.path.join(root, 'work')
    dop = os.path.join(root, 'doppler_traces')
    common = ['--train', 'S1', '--test', 'S2,S3', '--doppler_dir', dop, '--processed_dir', proc,
              '--work_dir', work, '--epochs', '10', '--window', '200', '--stride', '20',
              '--antennas', 'antenna1,antenna2', '--noise_level', '-3.0', '--despike']
    out = run('run_har.py', *common, '--loss', 'burst', '--loss_rate', '0,0.2,0.5',
              '--burst_ms', '100', cwd=root)
    loss_root = dop + '_loss'
    check(os.path.isdir(os.path.join(loss_root, 'none')), '欠落なしの基準を doppler_traces_loss/none に作る')
    for r in ['0.2', '0.5']:
        d = os.path.join(loss_root, f'burst_r{r}_b100ms')
        files = sorted(f for f in os.listdir(d) if f.endswith('_stream_0.txt'))
        check(files and all(f.startswith(('S2_', 'S3_')) for f in files),
              f'欠落率 {r}: テストのセッションだけ欠落を入れて計算する ({len(files)} ファイル)')
    check(not os.path.isdir(os.path.join(loss_root, 'burst_r0_b100ms')), '欠落率 0 は基準を使う')
    check(out.count('学習 (S1) ===') == 1, '学習は欠落なしのデータで 1 回だけ')

    summaries = sorted(f for f in os.listdir(os.path.join(work, 'outputs')) if f.startswith('har_summary_'))
    with open(os.path.join(work, 'outputs', summaries[-1]), encoding='utf-8') as fp:
        summary = fp.read()
    print(summary.split('\n## 混同行列')[0])
    check('欠落の条件ごとの正解率' in summary, 'まとめに欠落の条件ごとの表がある')
    for cond in ['| なし |', '| burst 欠落率 0.2, 平均 100 ms |', '| burst 欠落率 0.5, 平均 100 ms |']:
        check(cond in summary, f'まとめに「{cond}」の行がある')
    acc = {}
    for line in summary.splitlines():
        if line.startswith('| なし |') or line.startswith('| burst '):
            cells = [c.strip() for c in line.strip('|').split('|')]
            if len(cells) == 5:
                acc[cells[0]] = float(cells[4])
    check(acc.get('なし', 0) > 0.9, f'欠落なしの正解率 {acc.get("なし")}')
    plots = os.listdir(os.path.join(work, 'plots'))
    check(any('_burst_r0.5_b100ms_max_merge.png' in p for p in plots), '混同行列の図は条件ごとに分かれる')

    out = run('run_har.py', *common, '--loss', 'sustained', '--loss_rate', '0.3', '--loss_on', 'both',
              cwd=root)
    check(out.count('学習 (S1) ===') == 2, '--loss_on both: 欠落ありのデータでも学習し直す')
    d = os.path.join(loss_root, 'sustained_r0.3')
    check(any(f.startswith('S1_') for f in os.listdir(d)), '--loss_on both: 学習のセッションにも欠落を入れる')

    r = subprocess.run([sys.executable, os.path.join(_CODE, 'run_har.py'), *common[:4],
                        '--doppler_dir', dop, '--processed_dir', os.path.join(root, 'nothing'),
                        '--work_dir', work, '--loss', 'random', '--loss_rate', '0.1'],
                       cwd=root, capture_output=True, text=True)
    check(r.returncode != 0 and 'processed_phase' in r.stdout,
          '位相サニタイゼーション済みの CSI が無ければ止まる')


def main():
    workdir = tempfile.mkdtemp(prefix='sharp_usrp_loss_test_')
    try:
        test_patterns()
        test_seed_and_antennas()
        test_fill()
        proc = make_processed(workdir)
        test_cli(workdir, proc)
        if importlib.util.find_spec('tensorflow') and importlib.util.find_spec('sklearn'):
            test_run_har(workdir, proc)
        else:
            print('\n(TensorFlow / scikit-learn が無いため run_har.py のテストは飛ばします)')
    except TestFailure as e:
        print(f'\n失敗: {e}')
        return 1
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    print('\nすべてのテストに合格しました。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
