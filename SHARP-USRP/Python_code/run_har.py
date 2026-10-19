"""
    活動認識を 1 コマンドで最後まで回す (手順6〜8、--csi を付ければ手順0〜8)

    学習とテストのセッションを指定するだけで、データセット作成 → 学習 →
    テスト → 混同行列の描画 → 結果のまとめ までを順に実行する。--csi を付けると、
    その前にデコード済み CSI からドップラーデータを作るところ (run_batch.py) から始める。

    例:
      # ドップラーデータ (doppler_traces/) ができている場合
      python run_har.py --train S1 --test S2,S3

      # デコード済み CSI から全部 (E:/Dataset_CSI/S1/W_antenna1.mat ... がある場合)
      python run_har.py --train S1 --test S2,S3 --csi E:/Dataset_CSI --noise_level -3.0 --despike

      # アンテナ 2 本と 1 本の両方で回して比べる
      python run_har.py --train S1 --test S2,S3 --antennas antenna1,antenna2 --antennas antenna1

    活動ラベルは --activities で指定しなければ、学習セッションのファイル名
    (4 文字目) から集める。run_har.py が知らないオプション (--noise_level,
    --despike など) は run_batch.py にそのまま渡す。

    どこかの工程が失敗したら、その時点で止めて失敗した工程を表示する。

    結果:
      outputs/har_summary_<日時>.md  正解率の一覧 (同じ内容を画面にも出す)
      plots/complete_different_*.png  混同行列
      har_<日時>.log                  全工程の画面出力

    Copyright (C) 2026 Heisuke Takeda
    Released under the GNU GPL v3 (see LICENSE).
"""

import argparse
import glob
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime

import numpy as np

from dataset_utility import detect_activities, parse_antennas

HERE = os.path.dirname(os.path.abspath(__file__))


class StepFailed(Exception):
    pass


class Runner:
    """各工程を子プロセスで実行し、画面とログの両方に出力を流す。"""

    def __init__(self, log_path):
        self.log = open(log_path, 'a', encoding='utf-8', buffering=1)
        self.env = dict(os.environ, PYTHONIOENCODING='utf-8')

    def say(self, s=''):
        print(s, flush=True)
        self.log.write(s + '\n')

    def run(self, title, script, *args):
        self.say(f'\n=== {title} ===')
        self.say('$ python ' + ' '.join([script, *map(str, args)]))
        t0 = time.time()
        cmd = [sys.executable, '-u', os.path.join(HERE, script), *map(str, args)]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding='utf-8', errors='replace', env=self.env)
        for line in p.stdout:
            print(line, end='', flush=True)
            self.log.write(line)
        p.wait()
        if p.returncode != 0:
            raise StepFailed(f'「{title}」が失敗しました ({script}, 終了コード {p.returncode})')
        self.say(f'({time.time() - t0:.0f} 秒)')


def load_metrics(prefix, ants, acts, subdirs):
    path = os.path.join('./outputs', f'{prefix}_{ants}_{acts}_{subdirs}.txt')
    with open(path, 'rb') as fp:
        return pickle.load(fp)


def summarize(results, activities, train, stamp):
    lines = [f'# 活動認識の結果 ({stamp})', '',
             f'- 学習: {train} (測定ごとに 学習60% / 検証20% / テスト20%)',
             f'- 活動: {", ".join(activities)}', '',
             '| アンテナ | 評価データ | 正解率 (統合) | 正解率 (アンテナ単体) | 平均 F 値 (統合) | 活動ごとの F 値 (統合) |',
             '|---|---|---|---|---|---|']
    for ants, rows in results:
        for label, m in rows:
            fs = ', '.join(f'{a} {f:.2f}' for a, f in zip(activities, m['fscore_max_merge']))
            lines.append(f'| {ants} | {label} | {m["accuracy_max_merge"]:.3f} | '
                         f'{m["accuracy_single"]:.3f} | {np.mean(m["fscore_max_merge"]):.3f} | {fs} |')
    lines += ['', '「統合」は同じ時間窓に対する各アンテナの予測をまとめた最終判定。',
              f'「{train} の取り分け」は学習と同じ測定から取り分けたテスト区間 (時間は重ならない)。']

    for ants, rows in results:
        label, m = rows[1]      # 別セッションでの評価 (まとめたもの)
        conf = m['conf_matrix_max_merge']
        lines += ['', f'## 混同行列: {ants}, {label} (統合, 行: 正解, 列: 予測)', '',
                  '| | ' + ' | '.join(activities) + ' |',
                  '|---' * (len(activities) + 1) + '|']
        for a, row in zip(activities, conf):
            lines.append(f'| **{a}** | ' + ' | '.join(str(v) for v in row) + ' |')
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--train', required=True, help='学習に使うセッション (カンマ区切り。例: S1)')
    parser.add_argument('--test', required=True, help='テストに使うセッション (カンマ区切り。例: S2,S3)')
    parser.add_argument('--activities', default=None,
                        help='活動ラベル (カンマ区切り)。既定は学習セッションのファイル名から集める')
    parser.add_argument('--antennas', action='append', default=None,
                        help='使うアンテナ (既定 antenna1,antenna2)。複数回指定すると、'
                             'それぞれで学習・テストして比べる')
    parser.add_argument('--csi', default=None,
                        help='セッションごとのフォルダ (<csi>/S1/*.mat) がある場所。'
                             '指定するとドップラーデータの作成 (run_batch.py) から始める')
    parser.add_argument('--doppler_dir', default=None,
                        help='ドップラーデータの場所 (既定 Python_code/doppler_traces)')
    parser.add_argument('--name', default='usrp', help='モデルなどのファイル名の先頭')
    parser.add_argument('--work_dir', default=None,
                        help='モデル・結果・図・ログの書き出し先 (既定 Python_code)')
    parser.add_argument('--epochs', type=int, default=25, help='最大エポック数 (既定 25)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--window', type=int, default=340,
                        help='学習の 1 サンプルの時間方向の長さ (ドップラーの時間窓の数, 既定 340)')
    parser.add_argument('--stride', type=int, default=30, help='サンプルをずらす幅 (既定 30)')
    # ドップラー計算と揃える必要があるもの (--csi のときは run_batch.py にも渡す)
    parser.add_argument('--sample_length', type=int, default=31, help='ドップラー計算の窓長 (既定 31)')
    parser.add_argument('--sliding', type=int, default=1, help='ドップラー計算のずらし幅 (既定 1)')
    parser.add_argument('--n_fft', type=int, default=100, help='ドップラービン数 (既定 100)')
    args, batch_args = parser.parse_known_args()

    if batch_args and args.csi is None:
        parser.error(f'{" ".join(batch_args)} は --csi を付けたとき (run_batch.py に渡すとき) だけ使えます')

    # 指定されたパスは実行した場所から解決し、各スクリプトは work_dir で動かす
    # (outputs/ や plots/ などをそこからの相対パスで書くため)
    args.doppler_dir = os.path.abspath(args.doppler_dir or os.path.join(HERE, 'doppler_traces'))
    if args.csi is not None:
        args.csi = os.path.abspath(args.csi)
    work_dir = os.path.abspath(args.work_dir or HERE)
    os.makedirs(work_dir, exist_ok=True)
    os.chdir(work_dir)
    stamp = f'{datetime.now():%Y%m%d_%H%M%S}'
    runner = Runner(f'har_{stamp}.log')
    train = args.train
    test_sessions = args.test.split(',')
    antennas_list = [','.join(parse_antennas(a)) for a in (args.antennas or ['antenna1,antenna2'])]
    t_all = time.time()

    try:
        if args.csi is not None:
            sessions = list(dict.fromkeys(train.split(',') + test_sessions))
            patterns = [os.path.join(args.csi, s, '*.mat') for s in sessions]
            runner.run('ドップラーデータの作成 (手順0〜5)', 'run_batch.py', *patterns,
                       '--prefix_parent', '--doppler_dir', args.doppler_dir,
                       '--sample_length', args.sample_length, '--sliding', args.sliding,
                       '--n_fft', args.n_fft, *batch_args)

        if args.activities:
            activities = args.activities.split(',')
        else:
            activities = detect_activities(args.doppler_dir, train.split(','))
            if not activities:
                raise StepFailed(f'{args.doppler_dir} に {train}_ で始まるドップラーデータがありません')
            runner.say(f'\n活動ラベル (ファイル名から): {",".join(activities)}')
        unseen = sorted(set(detect_activities(args.doppler_dir, test_sessions)) - set(activities))
        if unseen:
            runner.say(f'注意: テスト側にしか無い活動 {",".join(unseen)} は評価に含めません')
        acts = ','.join(activities)

        win = [args.sample_length, args.sliding, args.window, args.stride, acts]
        net = [args.n_fft, args.window, 1, args.batch_size]
        # まとめた評価に加え、テストセッションが複数ならセッションごとにも評価する
        test_sets = [args.test] + (test_sessions if len(test_sessions) > 1 else [])

        results = []
        for ants in antennas_list:
            runner.run(f'[{ants}] 学習用データセット ({train})', 'CSI_doppler_create_dataset_train.py',
                       args.doppler_dir, train, *win, ants)
            runner.run(f'[{ants}] テスト用データセット ({args.test})', 'CSI_doppler_create_dataset_test.py',
                       args.doppler_dir, args.test, *win, ants)
            runner.run(f'[{ants}] 学習 ({train})', 'CSI_network.py',
                       args.doppler_dir, train, *net, ants, args.name, acts, '--epochs', args.epochs)
            rows = [(f'{train} の取り分け', load_metrics('test', ants, acts, train))]
            for ts in test_sets:
                runner.run(f'[{ants}] テスト ({ts})', 'CSI_network_test.py',
                           args.doppler_dir, ts, *net, ants, args.name, acts)
                runner.run(f'[{ants}] 混同行列の描画 ({ts})', 'CSI_network_metrics_plot.py',
                           f'complete_different_{ants}_{acts}_{ts}', acts)
                rows.append((ts, load_metrics('complete_different', ants, acts, ts)))
            results.append((ants, rows))
    except StepFailed as e:
        runner.say(f'\n*** {e}')
        runner.say(f'ログ: {os.path.join(work_dir, runner.log.name)}')
        return 1

    # TensorFlow のキャッシュはデータセット全体の複製で大きい (実データで数 GB) ため消す
    for f in glob.glob(f'{args.name}_*_cache_*'):
        os.remove(f)

    summary = summarize(results, activities, train, stamp)
    os.makedirs('outputs', exist_ok=True)
    summary_path = os.path.join('outputs', f'har_summary_{stamp}.md')
    with open(summary_path, 'w', encoding='utf-8') as fp:
        fp.write(summary)
    runner.say('\n' + '=' * 68)
    runner.say(summary)
    runner.say(f'全体の所要時間: {(time.time() - t_all) / 60:.1f} 分')
    runner.say(f'まとめ: {os.path.join(work_dir, summary_path)}')
    runner.say(f'混同行列の図: {os.path.join(work_dir, "plots")}')
    runner.say(f'ログ: {os.path.join(work_dir, runner.log.name)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
