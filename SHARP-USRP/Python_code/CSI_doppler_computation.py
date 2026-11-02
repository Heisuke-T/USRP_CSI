"""
    ドップラースペクトル (ドップラーマップ) の計算

    位相サニタイゼーション済みの CFR に短時間フーリエ変換を掛け、時間 x
    ドップラー速度のマップを得る。アルゴリズムは SHARP オリジナルと同じ
    (ハン窓 -> FFT -> パワー -> サブキャリア方向に加算 -> 正規化と雑音床)。

    SHARP オリジナルからの変更点と、その理由:

      1. --bandwidth / --sub_band による部分帯域の切り出しを廃止した
         オリジナルの --bandwidth 20 は「80MHz で取得したデータから 20MHz
         幅を切り出す」オプションで、245 列 (80MHz 分) の配列を前提とした
         固定インデックスだった。本システムの入力はすでに 20MHz なので、
         これを指定すると HE20 (245 列) では 20MHz のさらに 1/4 を切り出す
         誤りになる (エラーは出ないので気付きにくい)。
         部分帯域を見たい場合は --subcarrier_range で明示する。

      2. Tc (パケット間隔) と fc (中心周波数) をハードコードしない
         オリジナルは Tc=6ms, fc=5GHz 固定。USRP は受動受信のため取得間隔が
         装置ごと・測定ごとに異なる。既定では手順3 が保存した実測の
         パケット時刻から Tc を求める。

      3. 非等間隔サンプリングへの対応 (--resample)
         STFT は等間隔サンプリングを前提とするが、USRP は AP が実際に送信した
         パケットしか捉えられないため間隔が揺れる。実測 (変動係数 0.15) を
         模した合成データで 3 通りを比較した結果:

           速度      理想    none   nudft  interp
           0.5 m/s  0.595   0.592   0.547   0.592
           1.0 m/s  0.330   0.323   0.303   0.309
           2.0 m/s  0.330   0.302   0.302   0.247
           3.0 m/s  0.330   0.277   0.306   0.165
           (数値は真の速度±4ビンへのパワー集中度。高いほどにじみが少ない)

         - 2 m/s 程度まで (人の歩行) は 'none' で理想とほぼ同等
         - 3 m/s 付近では 'none' が理想の 84% まで落ち 'nudft' が取り戻す
         - 線形補間 'interp' は高速側で明確に悪化する (低域通過として働くため)

         よって既定は 'none' (SHARP と同じ扱い)。走行など速い動きを見る場合は
         'nudft' を推奨する。検証は tests/test_doppler.py。

      4. 受信レートの低下を模擬できる (--loss, 本実装で追加)
         パケットの欠落 (random / burst / periodic / sustained) を入れてから
         計算する。詳細は packet_loss.py。欠落の実績は _meta.json の
         packet_loss に残る。

    Copyright (C) 2026 Heisuke Takeda
    Based on SHARP (C) 2022 Francesca Meneghello, GNU GPL v3.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import argparse
import json
import math as mt
import os
import pickle
from os import listdir, path

import numpy as np
import scipy.io as sio
from scipy.fftpack import fft, fftshift
from scipy.signal.windows import hann

import packet_loss

V_LIGHT = 3e8


def resample_uniform(csi_complex, time_sec, target_interval=None):
    """複素 CFR を等間隔の時間グリッドへ線形補間する。

    csi_complex: [パケット x サブキャリア]
    time_sec:    [パケット]  実測のパケット時刻 [s]

    戻り値: (補間後の CFR, 実際の間隔 Tc, 補間の統計)
    """
    order = np.argsort(time_sec)
    t = time_sec[order]
    x = csi_complex[order, :]

    # 同時刻のパケットがあると補間できないため除く
    keep = np.concatenate(([True], np.diff(t) > 0))
    t, x = t[keep], x[keep, :]

    dt = np.diff(t)
    if target_interval is None:
        target_interval = float(np.median(dt))

    n_out = int(np.floor((t[-1] - t[0]) / target_interval)) + 1
    t_uniform = t[0] + np.arange(n_out) * target_interval

    # 実部と虚部を別々に補間する (複素数の線形補間と等価)
    out = (np.apply_along_axis(lambda c: np.interp(t_uniform, t, c), 0, x.real)
           + 1j * np.apply_along_axis(lambda c: np.interp(t_uniform, t, c), 0, x.imag))

    stats = dict(
        n_in=int(t.size), n_out=int(n_out),
        target_interval_ms=target_interval * 1e3,
        median_interval_ms=float(np.median(dt)) * 1e3,
        min_interval_ms=float(dt.min()) * 1e3,
        max_interval_ms=float(dt.max()) * 1e3,
        cv=float(dt.std() / dt.mean()),
        max_gap_in_samples=float(dt.max() / target_interval),
    )
    return out, target_interval, stats


def _finalize(profiles, n_pkt, num_symbols, noise_lev, normalize=True):
    """窓ごとに最大値で割り、雑音床で切る (SHARP と同じ)。

    normalize=False なら割り算も切り捨てもせず、パワーのまま返す
    (--save_unnormalized の nowinnorm / raw 版)。
    """
    if not profiles:
        raise ValueError(
            f'窓長 {num_symbols} に対してパケット数 {n_pkt} が '
            f'足りません。--sample_length を小さくしてください。')
    arr = np.asarray(profiles, dtype=float)
    if not normalize:
        return arr
    return normalize_windows(arr, noise_lev)


def normalize_windows(arr, noise_lev):
    """窓ごとに最大値で割り、10^noise_lev 未満を 10^noise_lev にそろえる。"""
    arr = arr / np.max(arr, axis=1, keepdims=True)
    arr[arr < mt.pow(10, noise_lev)] = mt.pow(10, noise_lev)
    return arr


def despike_packets(csi_complex, n_sigmas=8.0, verbose=True):
    """外れ値パケットを前後の補間で置き換える。

    振幅が飛んだパケットや位相がずれたパケットが 1 つあるだけで、STFT では
    窓長ぶんの窓すべてに全速度域へ広がるエネルギーが乗る。ドップラーマップ上
    では「全速度域を貫く縦縞」として現れ、物理的な動きと紛らわしい。

    パケットごとの CFR が前後の中央値からどれだけ外れているかを見て、
    ロバストな尺度 (中央絶対偏差) で n_sigmas を超えるものを線形補間で置換する。
    SHARP の前処理にも hampel_filter が用意されているが呼ばれていない。

    前後どちらとの差も大きい「孤立した飛び」だけを対象にする。動きによる変化は
    連続的なのでこの条件を満たしにくい。閾値を下げすぎると速い動きを消すため、
    既定は 8 (実測データで 0.5% のパケットが該当) としている。
    """
    bad = despike_mask(csi_complex, n_sigmas)
    if not bad.any():
        return csi_complex, 0
    out = interp_packets(csi_complex, bad)
    if verbose:
        print(f'  外れ値パケットを {int(bad.sum())} 件補間 '
              f'(全速度域に広がる縦縞の原因)')
    return out, int(bad.sum())


def despike_mask(csi_complex, n_sigmas=8.0):
    """外れ値とみなすパケットの印 (True = 外れ値)。判定の中身は despike_packets と同じ。"""
    n = csi_complex.shape[0]
    if n < 5:
        return np.zeros(n, dtype=bool)

    # 隣接パケットとの差の大きさ (動きがあっても連続的に変わるので小さい)
    d = np.abs(np.diff(csi_complex, axis=0)).mean(axis=1)
    med = np.median(d)
    mad = np.median(np.abs(d - med)) * 1.4826
    bad = np.zeros(n, dtype=bool)
    if mad <= 0:
        return bad

    # 前後どちらとの差も大きいパケットを外れ値とみなす
    jump = (d - med) / mad > n_sigmas
    bad[1:-1] = jump[:-1] & jump[1:]
    return bad


def interp_packets(csi_complex, bad):
    """印の付いたパケットを前後のパケットから線形補間で置き換える。"""
    out = csi_complex.copy()
    idx = np.arange(csi_complex.shape[0])
    good = ~bad
    for k in range(out.shape[1]):
        out[bad, k] = (np.interp(idx[bad], idx[good], csi_complex[good, k].real)
                       + 1j * np.interp(idx[bad], idx[good],
                                        csi_complex[good, k].imag))
    return out


def _strip_static(cut):
    """窓内の時間平均 (= 静止経路) を各サブキャリアから引く。

    直接波や家具などの静止経路は動かないので CFR の時間平均に集まる。
    実測では動きによる成分が静止成分より 11 dB ほど小さく、正規化と
    雑音床の切り捨てで潰れてしまう。平均を引くとこの比が 15 dB 改善する。
    """
    return cut - cut.mean(axis=0, keepdims=True)


def doppler_spectrum(csi_complex, num_symbols, sliding, noise_lev, n_fft=100,
                     remove_static=False, normalize=True):
    """短時間フーリエ変換でドップラースペクトルを得る (SHARP と同じ手順)。"""
    profiles = []
    hann_window = np.expand_dims(hann(num_symbols), axis=-1)

    for i in range(0, csi_complex.shape[0] - num_symbols, sliding):
        cut = np.nan_to_num(csi_complex[i:i + num_symbols, :])
        if remove_static:
            cut = _strip_static(cut)
        wind = np.multiply(cut, hann_window)
        prof = fftshift(fft(wind, n=n_fft, axis=0), axes=0)
        # パワーをサブキャリア方向に加算
        profiles.append(np.sum(np.abs(prof * np.conj(prof)), axis=1))

    return _finalize(profiles, csi_complex.shape[0], num_symbols, noise_lev,
                     normalize=normalize)


def doppler_spectrum_nudft(csi_complex, time_sec, num_symbols, sliding,
                           noise_lev, Tc, n_fft=100, remove_static=False,
                           normalize=True):
    """非等間隔DFT によるドップラースペクトル。

    実測のパケット時刻をそのまま指数の中に入れるため、補間を伴わずに
    非等間隔サンプリングを扱える。周波数グリッドは等間隔 FFT と同じ
    (中心 0、間隔 1/(Tc*n_fft)) に取るので速度軸の意味は変わらない。

    等間隔サンプリングであれば通常の FFT と一致する。
    """
    f_grid = (np.arange(n_fft) - n_fft // 2) / (Tc * n_fft)
    profiles = []

    for i in range(0, csi_complex.shape[0] - num_symbols, sliding):
        x = np.nan_to_num(csi_complex[i:i + num_symbols, :])
        if remove_static:
            x = _strip_static(x)
        t = time_sec[i:i + num_symbols]
        t = t - t[0]
        span = t[-1] if t[-1] > 0 else 1.0

        # ハン窓は標本番号ではなく時間位置で与える
        w = 0.5 * (1 - np.cos(2 * np.pi * t / span))
        # 各標本が代表する時間幅で重み付けする (標本密度の偏りの補正)
        dt = np.gradient(t)
        w = w * dt / dt.mean()

        prof = np.exp(-1j * 2 * np.pi * np.outer(f_grid, t)) @ (x * w[:, None])
        profiles.append(np.sum(np.abs(prof) ** 2, axis=1))

    return _finalize(profiles, csi_complex.shape[0], num_symbols, noise_lev,
                     normalize=normalize)


# --save_unnormalized で追加で保存する版 (名前: 説明)
VARIANTS = {
    'nowinnorm': 'パケットごとの振幅の割り算 (1) はする / 窓ごとの最大値での割り算と雑音床 (2) はしない',
    'raw': '(1) も (2) もしない (受信した振幅のまま)',
}


def variant_dir(doppler_root, vname, sub=None):
    """版ごとの保存先: <ドップラーの保存先>_<版の名前>/ (例 doppler_traces_raw/)。"""
    root = path.normpath(doppler_root) + '_' + vname
    return path.join(root, sub) if sub else root


def _window_centers(time_sec, sample_length, sliding, n_windows):
    """各窓の実時刻 (窓の中央、最初のパケット = 0)。作れなければ None。"""
    if time_sec is None or time_sec.size < sample_length:
        return None, None
    ts = np.sort(time_sec)
    centers = np.array([
        ts[i:i + sample_length].mean()
        for i in range(0, ts.size - sample_length, sliding)])
    if centers.size < n_windows:
        return None, ts
    return centers[:n_windows] - ts[0], ts


def process_one(mat_file, out_file, args, variant_files=None):
    """1 ファイルのドップラーを計算して保存する。

    args.save_unnormalized が真なら、正規化を省いた 2 つの版も保存する
    (variant_files = {版の名前: 保存先の .txt}。省略時は out_file のフォルダ名に
    _nowinnorm / _raw を付けたフォルダ)。どの版も同じパケット・同じ窓から作る。
    """
    mdic = sio.loadmat(mat_file)
    csi_matrix_processed = mdic['csi_matrix_processed']

    end = csi_matrix_processed.shape[0] - args.end if args.end > 0 else None
    csi_matrix_processed = csi_matrix_processed[args.start:end, :, :]

    time_sec = None
    if 'time_sec' in mdic:
        t = np.asarray(mdic['time_sec'], dtype=float).ravel()
        if t.size >= csi_matrix_processed.shape[0] + args.start:
            time_sec = t[args.start:end]

    save_variants = bool(getattr(args, 'save_unnormalized', False))
    if save_variants and variant_files is None:
        d = path.dirname(path.abspath(out_file))
        variant_files = {v: path.join(variant_dir(d, v), path.basename(out_file))
                         for v in VARIANTS}

    # 振幅をパケットごとに正規化してから複素数へ戻す (SHARP と同じ。(1))
    amp_raw = csi_matrix_processed[:, :, 0]
    phasor = np.exp(1j * csi_matrix_processed[:, :, 1])
    amp = amp_raw / np.mean(amp_raw, axis=1, keepdims=True)
    csi_complex = amp * phasor
    # (1) をしない版。以後のパケット単位の処理は csi_complex とまったく同じにかける
    csi_raw = amp_raw * phasor if save_variants else None

    # --- 受信レートの低下 (パケット欠落) の模擬 ---
    # 実際に受信が欠けた状況を作るため、外れ値の除去や間引きより前に入れる
    loss = None
    if getattr(args, 'loss', None):
        keep, t_used = packet_loss.mask_for(
            path.basename(mat_file)[:-4], time_sec, args,
            n=csi_complex.shape[0], Tc=args.Tc)
        t_before = time_sec
        csi_complex, time_sec = packet_loss.apply_mask(
            csi_complex, t_before, keep, args.loss_fill)
        if csi_raw is not None:
            csi_raw, _ = packet_loss.apply_mask(csi_raw, t_before, keep, args.loss_fill)
        loss = packet_loss.loss_stats(t_used, keep)
        loss['setting'] = packet_loss.describe(args)
        loss['fill'] = args.loss_fill
        print(f'  欠落の模擬: {loss["setting"]}')
        packet_loss.print_stats(loss, indent='    ')

    # 部分帯域の切り出し (既定は全サブキャリア)
    if args.subcarrier_range is not None:
        lo, hi = args.subcarrier_range
        csi_complex = csi_complex[:, lo:hi]
        if csi_raw is not None:
            csi_raw = csi_raw[:, lo:hi]

    # --- 外れ値パケットの除去 ---
    # 判定は (1) の正規化をした CFR で行い (従来どおり)、同じパケットを両方で置き換える
    n_despiked = 0
    if args.despike:
        bad = despike_mask(csi_complex, n_sigmas=args.despike_sigmas)
        n_despiked = int(bad.sum())
        if n_despiked:
            csi_complex = interp_packets(csi_complex, bad)
            if csi_raw is not None:
                csi_raw = interp_packets(csi_raw, bad)
            print(f'  外れ値パケットを {n_despiked} 件補間 '
                  f'(全速度域に広がる縦縞の原因)')

    # --- パケットの間引き ---
    # 速度軸の範囲は v_max = c/(Tc*fc)/2 なので、取得レートが高い測定ほど軸が
    # 広くなり、同じ窓長での速度分解能 c/(Tc*fc*窓長) も粗くなる。N 個に 1 個へ
    # 間引くと Tc が N 倍になり、レートの違う測定どうしで軸と分解能を揃えられる。
    # (SHARPax の --sub_sampling と同じ考え方)
    if args.sub_sampling > 1:
        n_before = csi_complex.shape[0]
        csi_complex = csi_complex[::args.sub_sampling, :]
        if csi_raw is not None:
            csi_raw = csi_raw[::args.sub_sampling, :]
        if time_sec is not None:
            time_sec = time_sec[::args.sub_sampling]
        print(f'  間引き 1/{args.sub_sampling}: '
              f'{n_before} -> {csi_complex.shape[0]} パケット')

    # --- サンプリング間隔 Tc を決める ---
    resample_stats = None
    Tc = args.Tc
    cv = None
    if time_sec is not None and time_sec.size == csi_complex.shape[0]:
        dt = np.diff(np.sort(time_sec))
        cv = float(dt.std() / dt.mean()) if dt.size else 0.0
        if Tc is None:
            Tc = float(np.median(dt))
        print(f'  パケット間隔: 中央値 {np.median(dt)*1e3:.3f} ms, '
              f'変動係数 {cv:.3f} -> Tc={Tc*1e3:.3f} ms')
        if cv > args.cv_warn_threshold and args.resample == 'none':
            print(f'  注意: 間隔のばらつきが大きめです (変動係数 {cv:.2f})。'
                  f'--resample nudft も試してください。')
    elif Tc is None:
        raise ValueError(
            'Tc を決められません。time_sec を含む .mat を使うか '
            '--Tc で明示してください。')

    # --- ドップラースペクトル ---
    # まず正規化しないパワーを求め、通常の版はそれを窓ごとに正規化して作る
    # (正規化の中身は SHARP と同じなので、通常の版の結果は従来と変わらない)
    def spectrum(x, quiet=False):
        if args.resample == 'nudft':
            if time_sec is None or time_sec.size != x.shape[0]:
                raise ValueError(f'{mat_file}: time_sec が無いため nudft は使えません。')
            order = np.argsort(time_sec)
            if not quiet:
                print('  非等間隔DFT で計算')
            return doppler_spectrum_nudft(
                x[order], time_sec[order], args.sample_length,
                args.sliding, args.noise_level, Tc, n_fft=args.n_fft,
                remove_static=args.remove_static, normalize=False), None
        if args.resample == 'interp':
            if time_sec is None or time_sec.size != x.shape[0]:
                raise ValueError(f'{mat_file}: time_sec が無いため interp は使えません。')
            xr, tc_rs, stats = resample_uniform(x, time_sec, args.Tc)
            if not quiet:
                print('  警告: 線形補間による等間隔化は高いドップラー速度でスペクトルを'
                      'にじませます。比較目的以外では nudft か none を推奨します。')
                print(f'  等間隔化: {stats["n_in"]} -> '
                      f'{stats["n_out"]} サンプル, Tc={tc_rs*1e3:.3f} ms')
            return doppler_spectrum(xr, args.sample_length, args.sliding,
                                    args.noise_level, n_fft=args.n_fft,
                                    remove_static=args.remove_static,
                                    normalize=False), (tc_rs, stats)
        return doppler_spectrum(x, args.sample_length, args.sliding,
                                args.noise_level, n_fft=args.n_fft,
                                remove_static=args.remove_static,
                                normalize=False), None

    power, rs = spectrum(csi_complex)
    if rs is not None:
        Tc, resample_stats = rs
    arr = normalize_windows(power, args.noise_level)
    power_raw = spectrum(csi_raw, quiet=True)[0] if csi_raw is not None else None

    # --- 速度軸 ---
    # ビン間隔: FFT のビン数から決まる (ゼロ詰めを含む)
    delta_v_bin = V_LIGHT / (Tc * args.fc * args.n_fft)
    # 分解能: 窓長から決まる (ゼロ詰めでは向上しない)
    delta_v_res = V_LIGHT / (Tc * args.fc * args.sample_length)
    v_max = V_LIGHT / (Tc * args.fc) / 2      # 一意に測れる速度の上限

    os.makedirs(path.dirname(path.abspath(out_file)), exist_ok=True)
    with open(out_file, 'wb') as fp:
        pickle.dump(arr, fp)

    # 各窓の実時刻 (窓の中央) を保存する。パケット到着が不均一だと
    # 「窓番号 x ずらし x Tc」で合成した時間軸は実時間とずれていくため、
    # 描画側はこちらを使う。
    centers, ts = _window_centers(time_sec, args.sample_length, args.sliding, arr.shape[0])
    if centers is not None:
        np.save(out_file[:-4] + '_time.npy', centers)
        print(f'  実測の長さ: {ts[-1]-ts[0]:.1f} s '
              f'(Tc から合成すると {arr.shape[0]*args.sliding*Tc:.1f} s)')

    meta = dict(
        source=path.basename(mat_file),
        shape=list(arr.shape),
        Tc=Tc, fc=args.fc, n_fft=args.n_fft,
        sample_length=args.sample_length, sliding=args.sliding,
        noise_level=args.noise_level,
        delta_v_bin=delta_v_bin, delta_v_resolution=delta_v_res,
        v_max=v_max,
        n_subcarriers=int(csi_complex.shape[1]),
        resample_mode=args.resample,
        remove_static=bool(args.remove_static),
        sub_sampling=int(args.sub_sampling),
        despike=bool(args.despike),
        despike_sigmas=float(args.despike_sigmas),
        despiked_packets=n_despiked,
        start=int(args.start), end=int(args.end),
        interval_cv=cv,
        resample=resample_stats,
        packet_loss=loss,
        variant='normalized',
        normalization='(1) パケットごとの振幅の割り算 + (2) 窓ごとの最大値での割り算と雑音床 (SHARP と同じ)',
    )
    with open(out_file[:-4] + '_meta.json', 'w', encoding='utf-8') as fp:
        json.dump(meta, fp, ensure_ascii=False, indent=2)

    print(f'  {arr.shape} (時間窓 x ドップラービン) -> {out_file}')
    print(f'  速度軸: ±{v_max:.2f} m/s, ビン間隔 {delta_v_bin:.3f} m/s, '
          f'分解能 {delta_v_res:.3f} m/s')

    # --- 正規化を省いた版 (--save_unnormalized) ---
    if save_variants:
        for vname, data in (('nowinnorm', power), ('raw', power_raw)):
            vfile = variant_files[vname]
            os.makedirs(path.dirname(path.abspath(vfile)), exist_ok=True)
            with open(vfile, 'wb') as fp:
                pickle.dump(data, fp)
            if centers is not None:
                np.save(vfile[:-4] + '_time.npy', centers)
            vmeta = dict(meta, variant=vname, normalization=VARIANTS[vname],
                         noise_level=None,
                         units='|H|^2 をサブキャリア方向に足したパワー (任意単位。'
                               '雑音床での切り捨てなし)')
            with open(vfile[:-4] + '_meta.json', 'w', encoding='utf-8') as fp:
                json.dump(vmeta, fp, ensure_ascii=False, indent=2)
            print(f'  {vname} 版 -> {vfile}')
    return meta


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dir', help='位相サニタイゼーション結果のディレクトリ')
    parser.add_argument('subdirs', help='サブディレクトリ (カンマ区切り)。'
                                        'サブディレクトリを使っていない場合は "." ')
    parser.add_argument('dir_doppler', help='ドップラーデータの保存先')
    parser.add_argument('start', help='先頭から捨てるパケット数', type=int)
    parser.add_argument('end', help='末尾から捨てるパケット数', type=int)
    parser.add_argument('sample_length', help='1窓あたりのパケット数', type=int)
    parser.add_argument('sliding', help='窓をずらすパケット数', type=int)
    parser.add_argument('noise_level', help='雑音床 (10^x で切り捨て)', type=float)
    parser.add_argument('--fc', type=float, default=5.18e9,
                        help='中心周波数 [Hz] (既定 5.18e9 = Ch36)')
    parser.add_argument('--Tc', type=float, default=None,
                        help='パケット間隔 [s]。既定は time_sec から実測')
    parser.add_argument('--n_fft', type=int, default=100,
                        help='FFT のビン数 (既定 100, SHARP と同じ)')
    parser.add_argument('--resample', choices=['none', 'nudft', 'interp'],
                        default='none',
                        help='非等間隔サンプリングの扱い。'
                             'none: そのまま等間隔とみなす (既定, SHARP と同じ。'
                             '実測の変動係数 0.15 程度なら理想とほぼ同等)。'
                             'nudft: 実測時刻を使う非等間隔DFT (間隔が大きく乱れる場合)。'
                             'interp: 線形補間で等間隔化 (比較用。高速側で悪化する)')
    parser.add_argument('--cv_warn_threshold', type=float, default=0.3,
                        help='この変動係数を超えたら nudft を勧める警告を出す')
    parser.add_argument('--despike', action='store_true',
                        help='外れ値パケットを前後の補間で置き換える。'
                             '全速度域を貫く縦縞が出ている場合に使う')
    parser.add_argument('--despike_sigmas', type=float, default=8.0,
                        help='外れ値と判定する閾値 (中央絶対偏差の倍数, 既定 8)。'
                             '小さくすると多く除去するが、速い動きを消す恐れがある'
                             '(実測: 8 で 0.5%%, 5 で 2.1%% のパケットを置換)')
    parser.add_argument('--sub_sampling', type=int, default=1,
                        help='N 個に 1 個へ間引く (既定 1 = 間引かない)。'
                             '取得レートが高い測定の速度軸と分解能を、'
                             '他の測定に揃えるために使う')
    parser.add_argument('--remove_static', action='store_true',
                        help='窓内の時間平均 (静止経路) を引いてから変換する。'
                             '直接波が強く動きが埋もれる場合に使う。'
                             '0 m/s の静止成分は消えるので、SHARP の学習側に'
                             '渡すときは既定 (除去しない) と揃えること')
    parser.add_argument('--save_unnormalized', action='store_true',
                        help='正規化を省いた版も保存する (回帰で動きの大きさを使うため)。'
                             '<dir_doppler>_nowinnorm/ に窓ごとの割り算なし、'
                             '<dir_doppler>_raw/ にパケットごとの割り算も窓ごとの割り算もなし。'
                             '通常の版は今までどおり dir_doppler に保存する')
    parser.add_argument('--subcarrier_range', type=int, nargs=2, default=None,
                        metavar=('LO', 'HI'),
                        help='使用するサブキャリアの範囲 (既定は全部)')
    parser.add_argument('--only', default=None,
                        help='名前がこれで始まるファイルだけ処理する (カンマ区切り。例: S2_,S3_)')
    parser.add_argument('--overwrite', action='store_true',
                        help='処理済み (出力の .txt がある) でもやり直す')
    packet_loss.add_loss_args(parser)
    args = parser.parse_args()
    try:
        packet_loss.check_loss_args(args)
    except ValueError as e:
        parser.error(str(e))

    # 雑音床は 10^x として使う。x >= 0 だと 10^x >= 1 となり、窓ごとの
    # 最大値 1.0 を含む全ビンが切り捨てられて真っ平らな図になる。
    if args.noise_level >= 0:
        parser.error(
            f'noise_level は負の値で指定してください (指定値 {args.noise_level})。'
            f'10^{args.noise_level:g} = {10**args.noise_level:g} となり、'
            f'正規化後の全ビン (最大 1.0) が切り捨てられて一様な図になります。'
            f'例: -3.0 (= -30 dB)')

    for subdir in args.subdirs.split(','):
        exp_dir = path.join(args.dir, subdir) if subdir != '.' else args.dir
        out_dir = (path.join(args.dir_doppler, subdir) if subdir != '.'
                   else args.dir_doppler)
        os.makedirs(out_dir, exist_ok=True)

        names = [f[:-4] for f in sorted(listdir(exp_dir)) if f.endswith('.mat')]
        if args.only:
            prefixes = tuple(p for p in args.only.split(',') if p)
            names = [n for n in names if n.startswith(prefixes)]
        if not names:
            print(f'{exp_dir}: .mat がありません')
            continue

        sub = subdir if subdir != '.' else None
        for name in names:
            out_file = path.join(out_dir, name + '.txt')
            vfiles = ({v: path.join(variant_dir(args.dir_doppler, v, sub), name + '.txt')
                       for v in VARIANTS} if args.save_unnormalized else None)
            todo = [out_file] + (list(vfiles.values()) if vfiles else [])
            if all(path.exists(f) for f in todo) and not args.overwrite:
                print(f'{name}: 処理済みのためスキップ')
                continue
            print(name)
            process_one(path.join(exp_dir, name + '.mat'), out_file, args,
                        variant_files=vfiles)


if __name__ == '__main__':
    main()
