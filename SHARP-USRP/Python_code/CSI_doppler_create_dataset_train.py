"""
    学習用データセットの作成 (学習・検証・テストに分ける)

    ドップラーデータ (CSI_doppler_computation.py の出力) を測定ごとに時間方向
    で 60% / 20% / 20% に分け、それぞれを窓に切って保存する。分け方と窓の
    切り方は SHARP と同じ。

    ファイル名の約束:
      <dir>/S1_E_antenna1_..._stream_0.txt  または  <dir>/S1/S1_E_antenna1_..._stream_0.txt
        セッション名 (S1) で始まり、4 文字目 ('E') が活動ラベル、
        antenna1 / antenna2 がアンテナ。データセットは <dir>/S1/ に書く

    例 (S1 で学習。アンテナ 2 本):
      python CSI_doppler_create_dataset_train.py ./doppler_traces/ S1 31 1 340 30 E,W,R antenna1,antenna2
    アンテナ 1 本だけにするには最後を antenna1 に変える。本数 (2 なら
    antenna1,antenna2) でも指定できる。

    Copyright (C) 2026 Heisuke Takeda
    Based on SHARP (C) 2022 Francesca Meneghello, GNU GPL v3.
    Released under the GNU GPL v3 (see LICENSE).
"""

import argparse
import math as mt
import os
import sys

import numpy as np

from dataset_utility import (check_doppler_axes, convert_to_number,
                             create_windows_antennas, find_recordings,
                             load_recording, parse_antennas, session_dirs,
                             set_name, write_set)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dir', help='ドップラーデータのディレクトリ')
    parser.add_argument('subdirs', help='サブディレクトリ (カンマ区切り。例: S1)')
    parser.add_argument('sample_lengths', help='ドップラー計算の 1 窓あたりのパケット数', type=int)
    parser.add_argument('sliding', help='ドップラー計算で窓をずらしたパケット数', type=int)
    parser.add_argument('windows_length', help='学習の 1 サンプルあたりの時間方向の長さ', type=int)
    parser.add_argument('stride_lengths', help='サンプルをずらす幅', type=int)
    parser.add_argument('labels_activities', help='使う活動ラベル (カンマ区切り。例: E,W,R)')
    parser.add_argument('antennas', help='使うアンテナ (例: antenna1,antenna2 / antenna1)。本数でも可')
    args = parser.parse_args()

    labels_activities = args.labels_activities
    csi_label_dict = labels_activities.split(',')
    antennas = parse_antennas(args.antennas)
    num_packets = args.sample_lengths
    gap = mt.ceil(num_packets / args.sliding)   # 分割の境目で窓が重ならないよう空ける幅

    for subdir in args.subdirs.split(','):
        trace_dir, exp_dir = session_dirs(args.dir, subdir)
        print(f'{subdir}: {trace_dir}  (アンテナ: {",".join(antennas)})')

        recordings = find_recordings(trace_dir, subdir, csi_label_dict, antennas)
        if not recordings:
            sys.exit(f'エラー: {subdir}_ で始まる {",".join(antennas)} のドップラーデータ (.txt) が '
                     f'{exp_dir} にも {args.dir} にもありません')
        os.makedirs(exp_dir, exist_ok=True)

        csi_matrices, labels, all_metas = [], [], []
        for label, files in recordings:
            print('  ' + ' + '.join(os.path.basename(f) for f in files))
            csi_matrix, metas = load_recording(files)
            csi_matrices.append(csi_matrix)
            labels.append(convert_to_number(label, csi_label_dict))
            all_metas.extend(metas)
        check_doppler_axes(all_metas)

        missing = [a for a in csi_label_dict if a not in {r[0] for r in recordings}]
        if missing:
            print(f'  警告: 活動 {",".join(missing)} のデータがありません')

        csi_train, csi_val, csi_test = [], [], []
        for csi_matrix in csi_matrices:
            ll = csi_matrix.shape[2]
            train_len = int(np.floor(ll * 0.6))
            csi_train.append(csi_matrix[:, :, :train_len])

            start_val = train_len + gap
            val_len = int(np.floor(ll * 0.2))
            csi_val.append(csi_matrix[:, :, start_val:start_val + val_len])

            start_test = start_val + val_len + gap
            csi_test.append(csi_matrix[:, :, start_test:])

        for kind, csi_set in (('train', csi_train), ('val', csi_val), ('test', csi_test)):
            windows, labels_set, num_windows = create_windows_antennas(
                csi_set, labels, args.windows_length, args.stride_lengths, remove_mean=False)
            write_set(exp_dir, set_name(kind, antennas, labels_activities),
                      windows, labels_set, num_windows)


if __name__ == '__main__':
    main()
