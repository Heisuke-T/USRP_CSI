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

      # 受信レートの低下を模擬して、欠落率ごとの正解率を比べる (packet_loss.py)
      python run_har.py --train S1 --test S2,S3 --noise_level -3.0 --despike \
          --loss burst --loss_rate 0.1,0.2,0.4 --burst_ms 100

    --loss を付けると、位相サニタイゼーション済みの CSI (processed_phase/) から
    ドップラーデータを作り直す。まず欠落なしで全セッションを計算し (基準)、
    次に欠落率ごとにテストのセッションへ欠落を入れて計算する。学習は欠落なしの
    データで 1 回だけ行い (--loss_on both なら学習側にも同じ欠落を入れて欠落率
    ごとに学習する)、欠落率ごとにテストして正解率を並べる。ドップラー計算の
    設定 (--noise_level, --despike など) は基準と欠落ありの両方に同じものを使う。
    書き出し先は doppler_traces_loss/<設定>/ (例: burst_r0.2_b100ms)。

    活動ラベルは --activities で指定しなければ、学習セッションのファイル名
    (4 文字目) から集める。run_har.py が知らないオプション (--noise_level,
    --despike など) は run_batch.py にそのまま渡す。

    どこかの工程が失敗したら、その時点で止めて失敗した工程を表示する。

    結果:
      outputs/har_summary_<日時>.md  正解率の一覧 (同じ内容を画面にも出す)
      plots/complete_different_*.png  混同行列 (--loss のときは末尾に欠落の設定が付く)
      har_<日時>.log                  全工程の画面出力

    Copyright (C) 2026 Heisuke Takeda
    Released under the GNU GPL v3 (see LICENSE).
"""

import argparse
import glob
import json
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime

import numpy as np

import packet_loss
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


# ----------------------------------------------------------------------------
# 受信レートの低下 (--loss) のときに使うもの
# ----------------------------------------------------------------------------

def doppler_parser():
    """ドップラー計算の設定 (run_batch.py と同じ名前)。--loss のときに
    CSI_doppler_computation.py へ渡すため、run_har.py が知らないオプションから拾う。"""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--noise_level', type=float, default=-1.5)
    p.add_argument('--dop_start', type=int, default=10)
    p.add_argument('--dop_end', type=int, default=10)
    p.add_argument('--fc', type=float, default=5.18e9)
    p.add_argument('--Tc', type=float, default=None)
    p.add_argument('--resample', choices=['none', 'nudft', 'interp'], default='none')
    p.add_argument('--despike', action='store_true')
    p.add_argument('--despike_sigmas', type=float, default=8.0)
    p.add_argument('--sub_sampling', type=int, default=1)
    p.add_argument('--remove_static', action='store_true')
    return p


def doppler_flags(d, n_fft):
    flags = ['--n_fft', n_fft, '--fc', d.fc, '--resample', d.resample,
             '--despike_sigmas', d.despike_sigmas, '--sub_sampling', d.sub_sampling]
    if d.Tc is not None:
        flags += ['--Tc', d.Tc]
    if d.despike:
        flags.append('--despike')
    if d.remove_static:
        flags.append('--remove_static')
    return flags


def processed_source(processed_dir, session):
    """セッションの位相サニタイゼーション済み CSI の場所 (サブディレクトリか直下か)。

    戻り値: (CSI_doppler_computation.py に渡す subdirs, ファイルの数)
    """
    sub = os.path.join(processed_dir, session)
    for subdir, d in ((session, sub), ('.', processed_dir)):
        if os.path.isdir(d):
            n = len(glob.glob(os.path.join(d, f'{session}_*.mat')))
            if n:
                return subdir, n
    return None, 0


def read_metas(doppler_dir, sessions):
    metas = []
    for s in sessions:
        for d in (os.path.join(doppler_dir, s), doppler_dir):
            for f in sorted(glob.glob(os.path.join(d, f'{s}_*_meta.json'))):
                with open(f, encoding='utf-8') as fp:
                    metas.append(json.load(fp))
    return metas


def compare_with_existing(doppler_dir, sessions, d, sample_length, sliding, n_fft):
    """既存のドップラーデータ (以前の実行) と設定が違えば警告の文を返す。"""
    metas = read_metas(doppler_dir, sessions)
    if not metas:
        return []
    m = metas[0]
    now = dict(noise_level=d.noise_level, remove_static=d.remove_static,
               sub_sampling=d.sub_sampling, resample_mode=d.resample, despike=d.despike,
               sample_length=sample_length, sliding=sliding, n_fft=n_fft)
    diff = [f'{k}: 以前 {m[k]} / 今回 {v}' for k, v in now.items() if k in m and m[k] != v]
    if 'despike' not in m and m.get('despiked_packets', 0) > 0 and not d.despike:
        diff.append('despike: 以前は外れ値の補間あり / 今回 なし')
    return diff


def loss_summary(doppler_dir, sessions):
    """テストのセッションに入れた欠落の実績 (ファイル全体の合計)。"""
    sts = [m['packet_loss'] for m in read_metas(doppler_dir, sessions) if m.get('packet_loss')]
    if not sts:
        return None
    n = sum(s['n_packets'] for s in sts)
    lost = sum(s['n_lost'] for s in sts)
    pps = [(s['rate_before_pps'], s['rate_after_pps']) for s in sts if s.get('rate_before_pps')]
    gaps = [s['max_gap_ms'] for s in sts if 'max_gap_ms' in s]
    return dict(loss_rate=lost / n if n else 0.0,
                pps_before=float(np.mean([a for a, _ in pps])) if pps else None,
                pps_after=float(np.mean([b for _, b in pps])) if pps else None,
                max_gap_ms=max(gaps) if gaps else None)


# ----------------------------------------------------------------------------
# 結果のまとめ
# ----------------------------------------------------------------------------

def summarize(results, activities, train, stamp, test, loss=None):
    """results: [(アンテナ, [(条件, 評価データ, 指標), ...]), ...]

    loss: --loss のときの情報 (dict)。None なら従来どおりの表。
    """
    lines = [f'# 活動認識の結果 ({stamp})', '',
             f'- 学習: {train} (測定ごとに 学習60% / 検証20% / テスト20%)',
             f'- 活動: {", ".join(activities)}']
    if loss:
        lines += [f'- 受信レートの低下の模擬: {loss["pattern"]} '
                  f'(欠落を入れたのは {loss["target"]}、欠落したパケットは {loss["fill"]})',
                  f'- ドップラーデータ: {loss["root"]}']
    lines.append('')

    if loss:
        ants_list = [a for a, _ in results]
        lines += [f'## 欠落の条件ごとの正解率 (評価: {test}, アンテナ統合)', '',
                  '| 欠落 | 実際の欠落率 | 受信レート [pkt/s] | 最長の途切れ [ms] | '
                  + ' | '.join(ants_list) + ' |',
                  '|---' * (4 + len(ants_list)) + '|']
        for cond, info in loss['conditions']:
            accs = []
            for _, rows in results:
                m = next((m for c, lab, m in rows if c == cond and lab == test), None)
                accs.append('-' if m is None else f'{m["accuracy_max_merge"]:.3f}')
            if info:
                pps = (f'{info["pps_before"]:.0f} → {info["pps_after"]:.0f}'
                       if info['pps_before'] else '-')
                gap = f'{info["max_gap_ms"]:.0f}' if info['max_gap_ms'] is not None else '-'
                lr = f'{info["loss_rate"]:.3f}'
            else:
                lr, pps, gap = '0', '-', '-'
            lines.append(f'| {cond} | {lr} | {pps} | {gap} | ' + ' | '.join(accs) + ' |')
        lines.append('')

    head = '| アンテナ | ' + ('欠落 | ' if loss else '') + \
        '評価データ | 正解率 (統合) | 正解率 (アンテナ単体) | 平均 F 値 (統合) | 活動ごとの F 値 (統合) |'
    lines += [head, '|---' * head.count(' | ') + '|---|']
    for ants, rows in results:
        for cond, label, m in rows:
            fs = ', '.join(f'{a} {f:.2f}' for a, f in zip(activities, m['fscore_max_merge']))
            lines.append(f'| {ants} | ' + (f'{cond} | ' if loss else '') +
                         f'{label} | {m["accuracy_max_merge"]:.3f} | '
                         f'{m["accuracy_single"]:.3f} | {np.mean(m["fscore_max_merge"]):.3f} | {fs} |')
    lines += ['', '「統合」は同じ時間窓に対する各アンテナの予測をまとめた最終判定。',
              f'「{train} の取り分け」は学習と同じ測定から取り分けたテスト区間 (時間は重ならない)。']

    for ants, rows in results:
        for cond, label, m in rows:
            if label != test:       # 別セッションでの評価 (まとめたもの) だけ載せる
                continue
            conf = m['conf_matrix_max_merge']
            title = f'{ants}, {label}' + (f', 欠落 {cond}' if loss else '')
            lines += ['', f'## 混同行列: {title} (統合, 行: 正解, 列: 予測)', '',
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
    # 受信レートの低下の模擬
    g = packet_loss.add_loss_args(parser, sweep=True)
    g.add_argument('--loss_on', choices=['test', 'both'], default='test',
                   help='欠落を入れる側。test: テストのセッションだけ (既定、学習は欠落なしで 1 回) / '
                        'both: 学習のセッションにも入れ、条件ごとに学習し直す')
    g.add_argument('--processed_dir', default=None,
                   help='位相サニタイゼーション済みの CSI の場所 (既定 Python_code/processed_phase)')
    g.add_argument('--loss_dir', default=None,
                   help='欠落を入れたドップラーデータの書き出し先 (既定 <doppler_dir>_loss)')
    args, batch_args = parser.parse_known_args()

    # --loss のときは、ドップラー計算の設定 (--noise_level など) を拾って使う
    dop, unknown = doppler_parser().parse_known_args(batch_args)
    if args.csi is None:
        if args.loss is None and batch_args:
            parser.error(f'{" ".join(batch_args)} は --csi を付けたとき (run_batch.py に渡すとき) '
                         'か --loss を付けたときだけ使えます')
        if unknown:
            parser.error(f'知らないオプションです: {" ".join(unknown)}')

    rates = [None]
    if args.loss is not None:
        try:
            if args.loss_rate is not None:
                rates = [float(r) for r in args.loss_rate.split(',') if r.strip()]
            for r in rates:
                packet_loss.check_loss_args(args, r)
        except ValueError as e:
            parser.error(str(e))
    elif args.loss_rate is not None:
        parser.error('--loss_rate は --loss (欠落のパターン) と一緒に指定してください')

    # 指定されたパスは実行した場所から解決し、各スクリプトは work_dir で動かす
    # (outputs/ や plots/ などをそこからの相対パスで書くため)
    args.doppler_dir = os.path.abspath(args.doppler_dir or os.path.join(HERE, 'doppler_traces'))
    processed_dir = os.path.abspath(args.processed_dir or os.path.join(HERE, 'processed_phase'))
    loss_root = os.path.abspath(args.loss_dir or args.doppler_dir.rstrip('/\\') + '_loss')
    if args.csi is not None:
        args.csi = os.path.abspath(args.csi)
    work_dir = os.path.abspath(args.work_dir or HERE)
    os.makedirs(work_dir, exist_ok=True)
    os.chdir(work_dir)
    stamp = f'{datetime.now():%Y%m%d_%H%M%S}'
    runner = Runner(f'har_{stamp}.log')
    train = args.train
    train_sessions = train.split(',')
    test_sessions = args.test.split(',')
    antennas_list = [','.join(parse_antennas(a)) for a in (args.antennas or ['antenna1,antenna2'])]
    t_all = time.time()

    # 条件: (表に載せる名前, 学習データの場所, テストデータの場所, 図の名前の末尾)
    conditions = [(None, args.doppler_dir, args.doppler_dir, '')]
    loss_info = None

    try:
        if args.csi is not None:
            sessions = list(dict.fromkeys(train_sessions + test_sessions))
            patterns = [os.path.join(args.csi, s, '*.mat') for s in sessions]
            runner.run('ドップラーデータの作成 (手順0〜5)', 'run_batch.py', *patterns,
                       '--prefix_parent', '--doppler_dir', args.doppler_dir,
                       '--sample_length', args.sample_length, '--sliding', args.sliding,
                       '--n_fft', args.n_fft, *batch_args)

        if args.loss is not None:
            conditions, loss_info = prepare_loss(runner, args, dop, rates, processed_dir,
                                                 loss_root, train_sessions, test_sessions)

        first_train_dir = conditions[0][1]
        if args.activities:
            activities = args.activities.split(',')
        else:
            activities = detect_activities(first_train_dir, train_sessions)
            if not activities:
                raise StepFailed(f'{first_train_dir} に {train}_ で始まるドップラーデータがありません')
            runner.say(f'\n活動ラベル (ファイル名から): {",".join(activities)}')
        unseen = sorted(set(detect_activities(conditions[0][2], test_sessions)) - set(activities))
        if unseen:
            runner.say(f'注意: テスト側にしか無い活動 {",".join(unseen)} は評価に含めません')
        acts = ','.join(activities)

        win = [args.sample_length, args.sliding, args.window, args.stride, acts]
        net = [args.n_fft, args.window, 1, args.batch_size]
        # まとめた評価に加え、テストセッションが複数ならセッションごとにも評価する
        test_sets = [args.test] + (test_sessions if len(test_sessions) > 1 else [])

        results = []
        for ants in antennas_list:
            rows, trained_on = [], None
            for cond, train_dir, test_dir, suffix in conditions:
                at = f'[{ants}]' + (f' [欠落 {cond}]' if cond else '')
                if train_dir != trained_on:
                    runner.run(f'{at} 学習用データセット ({train})', 'CSI_doppler_create_dataset_train.py',
                               train_dir, train, *win, ants)
                    runner.run(f'{at} 学習 ({train})', 'CSI_network.py',
                               train_dir, train, *net, ants, args.name, acts, '--epochs', args.epochs)
                    rows.append((cond, f'{train} の取り分け', load_metrics('test', ants, acts, train)))
                    trained_on = train_dir
                runner.run(f'{at} テスト用データセット ({args.test})', 'CSI_doppler_create_dataset_test.py',
                           test_dir, args.test, *win, ants)
                for ts in test_sets:
                    runner.run(f'{at} テスト ({ts})', 'CSI_network_test.py',
                               test_dir, ts, *net, ants, args.name, acts)
                    runner.run(f'{at} 混同行列の描画 ({ts})', 'CSI_network_metrics_plot.py',
                               f'complete_different_{ants}_{acts}_{ts}', acts, '--suffix', suffix)
                    rows.append((cond, ts, load_metrics('complete_different', ants, acts, ts)))
            results.append((ants, rows))
    except StepFailed as e:
        runner.say(f'\n*** {e}')
        runner.say(f'ログ: {os.path.join(work_dir, runner.log.name)}')
        return 1

    # TensorFlow のキャッシュはデータセット全体の複製で大きい (実データで数 GB) ため消す
    for f in glob.glob(f'{args.name}_*_cache_*'):
        os.remove(f)

    summary = summarize(results, activities, train, stamp, args.test, loss=loss_info)
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


def prepare_loss(runner, args, dop, rates, processed_dir, loss_root, train_sessions, test_sessions):
    """欠落なし (基準) と、欠落の条件ごとのドップラーデータを作る。

    戻り値: (条件のリスト, まとめ用の情報)
    """
    sessions = list(dict.fromkeys(train_sessions + test_sessions))
    sources = {}
    for s in sessions:
        subdir, n = processed_source(processed_dir, s)
        if not n:
            raise StepFailed(f'{processed_dir} に {s}_ で始まる .mat がありません。'
                             f'欠落の模擬には位相サニタイゼーション済みの CSI (processed_phase) '
                             f'が必要です (--processed_dir で場所を指定できます)')
        sources[s] = subdir

    diff = compare_with_existing(args.doppler_dir, train_sessions, dop,
                                 args.sample_length, args.sliding, args.n_fft)
    if diff:
        runner.say('\n注意: 以前のドップラーデータとドップラー計算の設定が違います '
                   '(--noise_level, --despike などを以前と揃えるか確かめてください)')
        for d in diff:
            runner.say(f'  {d}')

    positional = [args.sample_length, args.sliding, dop.noise_level]
    flags = doppler_flags(dop, args.n_fft)

    def compute(out_dir, which, title, loss_flags):
        for s in which:
            runner.run(f'{title} ({s})', 'CSI_doppler_computation.py',
                       processed_dir, sources[s], out_dir, dop.dop_start, dop.dop_end, *positional,
                       '--only', f'{s}_', '--overwrite', *flags, *loss_flags)

    clean_dir = os.path.join(loss_root, 'none')
    compute(clean_dir, sessions, 'ドップラー計算: 欠落なし', [])
    conditions = [('なし', clean_dir, clean_dir, '_none')]
    infos = [('なし', None)]

    for r in rates:
        if r is not None and r == 0:
            continue        # 欠落率 0 は基準と同じ
        t = packet_loss.tag(args, r)
        d = os.path.join(loss_root, t)
        cond = packet_loss.describe(args, r)
        which = sessions if args.loss_on == 'both' else test_sessions
        compute(d, which, f'ドップラー計算: 欠落 {cond}', packet_loss.loss_cli_args(args, r))
        conditions.append((cond, d if args.loss_on == 'both' else clean_dir, d, '_' + t))
        infos.append((cond, loss_summary(d, test_sessions)))

    info = dict(pattern=args.loss, root=loss_root, conditions=infos, fill=args.loss_fill,
                target='学習とテストの両方' if args.loss_on == 'both' else 'テストのセッションだけ')
    return conditions, info


if __name__ == '__main__':
    sys.exit(main())
