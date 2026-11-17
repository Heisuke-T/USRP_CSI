"""
    --save_unnormalized (正規化を省いた版の保存) の検証。

    位相サニタイゼーション後と同じ形の .mat (csi_matrix_processed, time_sec) を合成し、
    CSI_doppler_computation.process_one で
      1. 通常の版の出力が、この機能を足す前と同じであること
         (オプションを付けても付けなくても同じ / 窓ごとに正規化し直すと nowinnorm 版と一致)
      2. nowinnorm 版 = パケットごとの割り算だけした正規化前のパワー
      3. raw 版 = 振幅のまま。振幅を k 倍すると k^2 倍になり、nowinnorm 版は変わらない
      4. 外れ値の補間・欠落の模擬・間引き・nudft でも、3 つの版が同じ窓・同じ時刻になること
      5. 保存先のフォルダ (doppler_traces_nowinnorm/, doppler_traces_raw/) と _time.npy・_meta.json
    を確かめる。

    実行:  python3 tests/test_doppler_variants.py

    Copyright (C) 2026 Heisuke Takeda
    Released under the GNU GPL v3 (see LICENSE).
"""

import json
import os
import pickle
import shutil
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import scipy.io as sio

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import CSI_doppler_computation as cdc  # noqa: E402

FC = 5.18e9
TC = 6e-3
N_PKT, N_SUB = 900, 57


def make_mat(path, amp_scale=1.0, seed=0, spikes=True):
    """動く反射 1 本 + 静止経路の CFR を、位相サニタイゼーション後の形で保存する。"""
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.uniform(0.8, 1.2, N_PKT)) * TC
    f_sub = (np.arange(N_SUB) - N_SUB // 2) * 312.5e3
    v = 1.0 * np.sin(2 * np.pi * 0.8 * t)            # 足踏みのような往復運動 [m/s]
    phase_d = 2 * np.pi * FC / cdc.V_LIGHT * 2 * np.cumsum(v * np.gradient(t))
    H = (1.0 + 0.0j) + 0.3 * np.exp(1j * (phase_d[:, None] + 2 * np.pi * f_sub[None, :] * 20e-9))
    H = H * (1.0 + 0.2 * np.sin(2 * np.pi * 0.1 * t))[:, None]   # 全体の振幅がゆっくり変わる
    H = H + 0.01 * (rng.normal(size=H.shape) + 1j * rng.normal(size=H.shape))
    if spikes:
        H[[200, 555]] *= 4.0                          # 孤立した外れ値パケット
    H = H * amp_scale
    m = np.stack([np.abs(H), np.angle(H)], axis=-1)
    sio.savemat(path, {'csi_matrix_processed': m, 'time_sec': t[None, :]})
    return t


def args_for(**kw):
    a = SimpleNamespace(start=10, end=10, sample_length=31, sliding=1, noise_level=-3.0, fc=FC, Tc=None,
                        n_fft=100, resample='none', cv_warn_threshold=0.3, subcarrier_range=None,
                        remove_static=False, sub_sampling=1, despike=True, despike_sigmas=8.0,
                        save_unnormalized=False, loss=None)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def load(p):
    with open(p, 'rb') as fp:
        return pickle.load(fp)


def old_process(mat_file, args):
    """この機能を足す前の計算手順 (正規化済みの版) を、ここで同じ順に再現する。"""
    mdic = sio.loadmat(mat_file)
    m = mdic['csi_matrix_processed']
    end = m.shape[0] - args.end
    m = m[args.start:end]
    t = np.asarray(mdic['time_sec'], float).ravel()[args.start:end]
    amp = m[:, :, 0] / np.mean(m[:, :, 0], axis=1, keepdims=True)
    x = amp * np.exp(1j * m[:, :, 1])
    if args.despike:
        x, _ = cdc.despike_packets(x, args.despike_sigmas, verbose=False)
    if args.sub_sampling > 1:
        x, t = x[::args.sub_sampling], t[::args.sub_sampling]
    if args.resample == 'nudft':
        Tc = float(np.median(np.diff(np.sort(t))))
        o = np.argsort(t)
        return cdc.doppler_spectrum_nudft(x[o], t[o], args.sample_length, args.sliding, args.noise_level,
                                          Tc, n_fft=args.n_fft, remove_static=args.remove_static)
    return cdc.doppler_spectrum(x, args.sample_length, args.sliding, args.noise_level, n_fft=args.n_fft,
                                remove_static=args.remove_static)


def main():
    tmp = tempfile.mkdtemp(prefix='dopvar_')
    ok = True
    try:
        m1 = os.path.join(tmp, 'S1_W_antenna1_stream_0.mat')
        m3 = os.path.join(tmp, 'S1_W_antenna1x3_stream_0.mat')
        make_mat(m1, 1.0)
        make_mat(m3, 3.0)
        cases = [dict(), dict(remove_static=True), dict(resample='nudft'), dict(sub_sampling=2),
                 dict(loss='burst', loss_rate=0.2, burst_ms=100.0, loss_seed=1, loss_fill='drop')]
        for kw in cases:
            label = ', '.join(f'{k}={v}' for k, v in kw.items()) or '既定'
            base = os.path.join(tmp, 'doppler_traces')
            out = os.path.join(base, 'a.txt')
            a_plain = args_for(**kw)
            if 'loss' in kw:
                # fill in the other packet-loss options with their defaults / 欠落の模擬のほかの設定は既定値
                import argparse
                import packet_loss
                p = argparse.ArgumentParser()
                packet_loss.add_loss_args(p)
                for k, v in vars(p.parse_args([])).items():
                    if k not in kw:
                        setattr(a_plain, k, v)
            cdc.process_one(m1, out, a_plain)
            plain = load(out)
            # 1a: same as before the change (only when the old path can be reproduced here)
            if 'loss' not in kw:
                old = old_process(m1, a_plain)
                same_old = np.allclose(plain, old, rtol=1e-12, atol=0)
            else:
                same_old = True
            a_var = args_for(**vars(a_plain))
            a_var.save_unnormalized = True
            out2 = os.path.join(tmp, 'doppler_traces2', 'a.txt')
            vf = {v: os.path.join(cdc.variant_dir(os.path.join(tmp, 'doppler_traces2'), v), 'a.txt')
                  for v in cdc.VARIANTS}
            cdc.process_one(m1, out2, a_var, variant_files=vf)
            main_v = load(out2)
            nowin = load(vf['nowinnorm'])
            raw = load(vf['raw'])
            t_main = np.load(out2[:-4] + '_time.npy')
            t_now = np.load(vf['nowinnorm'][:-4] + '_time.npy')
            t_raw = np.load(vf['raw'][:-4] + '_time.npy')
            meta = json.load(open(vf['raw'][:-4] + '_meta.json', encoding='utf-8'))
            checks = {
                '通常の版が機能追加前と同じ': same_old,
                'オプションの有無で通常の版が同じ': np.array_equal(plain, main_v),
                'nowinnorm を正規化し直すと通常の版': np.allclose(cdc.normalize_windows(nowin, -3.0), main_v),
                '3 つの版の形が同じ': main_v.shape == nowin.shape == raw.shape,
                '3 つの版の時刻が同じ': np.array_equal(t_main, t_now) and np.array_equal(t_main, t_raw),
                'nowinnorm は正規化されていない': nowin.max() > 1.5,
                'meta に版の名前': meta.get('variant') == 'raw' and meta.get('noise_level') is None,
            }
            # 3: amplitude x3 -> raw x9, nowinnorm unchanged (no loss: same packets)
            if 'loss' not in kw:
                out3 = os.path.join(tmp, 'doppler_traces3', 'b.txt')
                vf3 = {v: os.path.join(cdc.variant_dir(os.path.join(tmp, 'doppler_traces3'), v), 'b.txt')
                       for v in cdc.VARIANTS}
                cdc.process_one(m3, out3, a_var, variant_files=vf3)
                checks['振幅 3 倍で raw 版は 9 倍'] = np.allclose(load(vf3['raw']), 9 * raw, rtol=1e-6)
                checks['振幅 3 倍で nowinnorm 版は同じ'] = np.allclose(load(vf3['nowinnorm']), nowin, rtol=1e-6)
            print(f'\n[{label}]')
            for k, v in checks.items():
                print(f'  {"OK " if v else "NG "} {k}')
                ok &= bool(v)
            shutil.rmtree(os.path.join(tmp, 'doppler_traces'), ignore_errors=True)

        # 5: default folders next to the output folder / 既定の保存先
        out = os.path.join(tmp, 'doppler_traces', 'W', 'c.txt')
        cdc.process_one(m1, out, args_for(save_unnormalized=True))
        exp = [os.path.join(tmp, 'doppler_traces', 'W_nowinnorm', 'c.txt'),
               os.path.join(tmp, 'doppler_traces', 'W_raw', 'c.txt')]
        print('\n[既定の保存先（variant_files を渡さないとき）]')
        for p in exp:
            e = os.path.exists(p)
            print(f'  {"OK " if e else "NG "} {os.path.relpath(p, tmp)}')
            ok &= e
        print('  ' + ('OK ' if cdc.variant_dir('./doppler_traces/', 'raw', 'W').replace('\\', '/')
                      == 'doppler_traces_raw/W' else 'NG ') + "variant_dir('./doppler_traces/', 'raw', 'W') = "
              + cdc.variant_dir('./doppler_traces/', 'raw', 'W'))
        ok &= cdc.variant_dir('./doppler_traces/', 'raw', 'W').replace('\\', '/') == 'doppler_traces_raw/W'
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print('\nすべてのテストに合格しました。' if ok else '\n不合格があります。')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
