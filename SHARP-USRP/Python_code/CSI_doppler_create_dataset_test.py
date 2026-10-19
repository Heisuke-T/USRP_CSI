"""
    テスト用データセットの作成 (測定の全区間を使う)

    学習に使わない測定 (別の日・別の環境など) を、分割せずに全区間ぶん窓に
    切って保存する。窓の切り方は SHARP と同じ。

    ファイル名の約束は CSI_doppler_create_dataset_train.py と同じ。

    例 (S2, S3 でテスト。アンテナ 2 本):
      python CSI_doppler_create_dataset_test.py ./doppler_traces/ S2,S3 31 1 340 30 E,W,R antenna1,antenna2

    SHARP はここで同じサブディレクトリの学習用データセットを消していたが、
    本実装は消さない (同じ測定を学習にもテストにも作れる)。

    Copyright (C) 2026 Heisuke Takeda
    Based on SHARP (C) 2022 Francesca Meneghello, GNU GPL v3.
    Released under the GNU GPL v3 (see LICENSE).
"""

import argparse
import os
import sys

from dataset_utility import (check_doppler_axes, convert_to_number,
                             create_windows_antennas, find_recordings,
                             load_recording, parse_antennas, session_dirs,
                             set_name, write_set)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dir', help='ドップラーデータのディレクトリ')
    parser.add_argument('subdirs', help='サブディレクトリ (カンマ区切り。例: S2,S3)')
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

    for subdir in args.subdirs.split(','):
        trace_dir, exp_dir = session_dirs(args.dir, subdir)
        print(f'{subdir}: {trace_dir}  (アンテナ: {",".join(antennas)})')

        recordings = find_recordings(trace_dir, subdir, csi_label_dict, antennas)
        if not recordings:
            sys.exit(f'エラー: {subdir}_ で始まる {",".join(antennas)} のドップラーデータ (.txt) が '
                     f'{exp_dir} にも {args.dir} にもありません')
        os.makedirs(exp_dir, exist_ok=True)

        csi_complete, labels, all_metas = [], [], []
        for label, files in recordings:
            print('  ' + ' + '.join(os.path.basename(f) for f in files))
            csi_matrix, metas = load_recording(files)
            csi_complete.append(csi_matrix)
            labels.append(convert_to_number(label, csi_label_dict))
            all_metas.extend(metas)
        check_doppler_axes(all_metas)

        missing = [a for a in csi_label_dict if a not in {r[0] for r in recordings}]
        if missing:
            print(f'  警告: 活動 {",".join(missing)} のデータがありません')

        windows, labels_wind, num_windows = create_windows_antennas(
            csi_complete, labels, args.windows_length, args.stride_lengths, remove_mean=False)
        write_set(exp_dir, set_name('complete', antennas, labels_activities),
                  windows, labels_wind, num_windows)


if __name__ == '__main__':
    main()
