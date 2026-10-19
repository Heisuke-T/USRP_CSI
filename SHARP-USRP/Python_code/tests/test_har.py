"""
    活動認識 (データセット作成 → 学習 → テスト) の自己診断テスト。

    活動ごとに違うドップラー速度にパワーが集まる合成ドップラーデータを、
    実データと同じ名前 (S1_E_antenna1_..._stream_0.txt と _meta.json) で
    S1〜S3 に作り、

      1. 活動ラベルをファイル名の 4 文字目から読むこと
      2. アンテナの指定 (2 本 / 1 本 / 本数) どおりに束ねること
         (長さの違う 2 台の受信機は短いほうに揃える、相方の無い測定は使わない)
      3. _meta.json を読み込み対象にしないこと
      4. S1 で学習し、S2,S3 で評価する一連のコマンドが通り、合成データを
         正しく分類できること (TensorFlow がある場合のみ)
      5. run_batch.py にセッションのフォルダを渡したときの置き方
         (doppler_traces/S1_W_antenna1_stream_0.txt) でも、run_har.py 1 本で
         最後まで回り、結果のまとめが出ること (TensorFlow がある場合のみ)

    を確認する。実行:  python3 tests/test_har.py

    Copyright (C) 2026 Heisuke Takeda
    Released under the GNU GPL v3 (see LICENSE).
"""

import importlib.util
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
sys.path.insert(0, _CODE)

from dataset_utility import parse_antennas  # noqa: E402

ACTIVITIES = ['E', 'W', 'R']
# 活動ごとのドップラーピークのビン (中央 50 が 0 m/s)
PEAKS = {'E': [50], 'W': [38, 62], 'R': [22, 78], 'J': [30, 70]}
LENGTHS = {'antenna1': 2400, 'antenna2': 2300}   # 2 台で受けたパケット数が違う
WINDOW, STRIDE = 340, 30


class TestFailure(AssertionError):
    pass


def check(cond, msg):
    if not cond:
        raise TestFailure(msg)
    print(f'  OK: {msg}')


def synth_doppler(act, n_rows, rng):
    """[時間窓 x 100 ビン] の正規化済みパワー (CSI_doppler_computation.py の出力と同じ形)。"""
    bins = np.arange(100)
    arr = 0.05 * rng.random((n_rows, 100))
    for p in PEAKS[act]:
        jitter = rng.normal(0, 2, size=(n_rows, 1))
        arr += np.exp(-0.5 * ((bins[None, :] - p - jitter) / 3.0) ** 2)
    arr = arr / arr.max(axis=1, keepdims=True)
    return np.maximum(arr, 10 ** -1.5)


def make_traces(root, flat=False):
    """flat=False: <root>/S1/S1_E_..., flat=True: <root>/S1_E_... (run_batch.py の 2 通りの出力)"""
    rng = np.random.default_rng(0)
    for sess in ['S1', 'S2', 'S3']:
        d = root if flat else os.path.join(root, sess)
        os.makedirs(d, exist_ok=True)
        for act in ACTIVITIES + ['J']:            # J は使わない活動
            for ant, n in LENGTHS.items():
                stem = f'{sess}_{act}_{ant}_WAX202_aligned_CSI_stream_0'
                with open(os.path.join(d, stem + '.txt'), 'wb') as fp:
                    pickle.dump(synth_doppler(act, n, rng), fp)
                with open(os.path.join(d, stem + '_meta.json'), 'w') as fp:
                    json.dump({'v_max': 5.0, 'Tc': 0.0058}, fp)
        # 相方 (antenna2) の無い測定
        stem = f'{sess}_W_antenna1_WAX202_extra_CSI_stream_0'
        with open(os.path.join(d, stem + '.txt'), 'wb') as fp:
            pickle.dump(synth_doppler('W', 2400, rng), fp)
        # ドップラー計算が横に出す実測時刻 (読み込み対象ではない)
        np.save(os.path.join(d, f'{sess}_E_antenna1_WAX202_aligned_CSI_stream_0_time.npy'),
                np.arange(10.0))


def run(script, *args, cwd):
    cmd = [sys.executable, os.path.join(_CODE, script), *args]
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        raise TestFailure(f'{script} が失敗しました')
    return r.stdout


def load(path):
    with open(path, 'rb') as fp:
        return pickle.load(fp)


def test_dataset(dt):
    print('\n--- データセット作成 ---')
    check(parse_antennas('2') == ['antenna1', 'antenna2'], '本数 2 は antenna1,antenna2 になる')
    check(parse_antennas('antenna1') == ['antenna1'], 'アンテナ名で 1 本だけ指定できる')

    acts = ','.join(ACTIVITIES)
    out = run('CSI_doppler_create_dataset_train.py', dt + '/', 'S1', '31', '1',
              str(WINDOW), str(STRIDE), acts, 'antenna1,antenna2', cwd=dt)
    check('extra' in out and 'antenna2 が無い' in out, '相方の無い測定は警告して使わない')
    check('2300 に揃えます' in out, '長さの違うアンテナは短いほうに揃える')

    s1 = os.path.join(dt, 'S1')
    labels = load(os.path.join(s1, f'labels_train_antenna1,antenna2_{acts}.txt'))
    files = load(os.path.join(s1, f'files_train_antenna1,antenna2_{acts}.txt'))
    check(sorted(set(labels)) == [0, 1, 2], f'ラベルは 4 文字目から E,W,R -> 0,1,2 ({sorted(set(labels))})')
    # 学習区間 = floor(2300*0.6) = 1380 → 窓数 len(range(0, 1380-340, 30)) = 35 / 活動
    check(len(labels) == 3 * 35, f'学習区間の窓数 ({len(labels)})')
    w = load(files[0])
    check(w.shape == (2, 100, WINDOW) and w.dtype == np.float32,
          f'窓の形は [アンテナ 2 x 100 x {WINDOW}] の float32 ({w.shape}, {w.dtype})')

    run('CSI_doppler_create_dataset_train.py', dt + '/', 'S1', '31', '1',
        str(WINDOW), str(STRIDE), acts, 'antenna1', cwd=dt)
    files1 = load(os.path.join(s1, f'files_train_antenna1_{acts}.txt'))
    w1 = load(files1[0])
    check(w1.shape == (1, 100, WINDOW), f'antenna1 だけなら [1 x 100 x {WINDOW}] ({w1.shape})')
    # antenna1 単独は 2400 行そのまま: floor(2400*0.6)=1440 → 37 窓
    check(len(files1) == 3 * 37 + 1 * 37, f'antenna1 単独では相方の無い測定も使う ({len(files1)})')
    check(os.path.exists(os.path.join(s1, f'train_antenna1,antenna2_{acts}')),
          '2 本用のデータセットは 1 本用で上書きされない')
    check(np.array_equal(w1[0], load(files[0])[0]), 'antenna1 の中身は 2 本用のデータセットの 1 本目と同じ')

    out = run('CSI_doppler_create_dataset_test.py', dt + '/', 'S2,S3', '31', '1',
              str(WINDOW), str(STRIDE), acts, '2', cwd=dt)
    for sess in ['S2', 'S3']:
        lab = load(os.path.join(dt, sess, f'labels_complete_antenna1,antenna2_{acts}.txt'))
        # 全区間 2300 行 → len(range(0, 2300-340, 30)) = 66 窓 / 活動
        check(len(lab) == 3 * 66, f'{sess}: テスト用は全区間を使う ({len(lab)} 窓)')
    check(os.path.exists(os.path.join(s1, f'train_antenna1,antenna2_{acts}')),
          'テスト用の作成で S1 の学習用データセットは消えない')


def test_network(dt):
    print('\n--- 学習 (S1) とテスト (S2,S3) ---')
    acts = ','.join(ACTIVITIES)
    out = run('CSI_network.py', dt + '/', 'S1', '100', str(WINDOW), '1', '32',
              'antenna1,antenna2', 'usrp', acts, '--epochs', '8', cwd=dt)
    check(os.path.exists(os.path.join(dt, f'usrp_antenna1,antenna2_{acts}_network.keras')),
          'モデルが保存される')

    out = run('CSI_network_test.py', dt + '/', 'S2,S3', '100', str(WINDOW), '1', '32',
              'antenna1,antenna2', 'usrp', acts, cwd=dt)
    name = f'complete_different_antenna1,antenna2_{acts}_S2,S3'
    m = load(os.path.join(dt, 'outputs', name + '.txt'))
    check(m['conf_matrix_max_merge'].shape == (3, 3), '混同行列は 3 クラス')
    check(m['accuracy_max_merge'] > 0.9,
          f'S2,S3 での正解率 (アンテナ統合) {m["accuracy_max_merge"]:.3f}')
    ant = load(os.path.join(dt, 'outputs', 'change_number_antennas_' + name + '.txt'))
    check(ant['average_accuracy_change_num_ant'].shape == (2,), 'アンテナ 1 本・2 本の両方で評価される')

    out = run('CSI_network_metrics.py', name, acts, cwd=dt)
    check('使うアンテナの本数ごと' in out, '指標を表示できる')

    # 2 本で学習したモデルを antenna1 だけで評価する
    run('CSI_doppler_create_dataset_test.py', dt + '/', 'S2', '31', '1',
        str(WINDOW), str(STRIDE), acts, 'antenna1', cwd=dt)
    run('CSI_network_test.py', dt + '/', 'S2', '100', str(WINDOW), '1', '32',
        'antenna1', 'usrp', acts, '--model', f'usrp_antenna1,antenna2_{acts}_network.keras', cwd=dt)
    m1 = load(os.path.join(dt, 'outputs', f'complete_different_antenna1_{acts}_S2.txt'))
    check(m1['accuracy_max_merge'] > 0.9, f'antenna1 だけでの正解率 {m1["accuracy_max_merge"]:.3f}')

    if importlib.util.find_spec('matplotlib'):
        run('CSI_network_metrics_plot.py', name, acts, cwd=dt)
        check(os.path.exists(os.path.join(dt, 'plots', name + '_max_merge.png')), '混同行列を描画できる')


def test_batch_names():
    print('\n--- run_batch.py の出力名 ---')
    if not (importlib.util.find_spec('scipy') and importlib.util.find_spec('h5py')):
        print('  (scipy / h5py が無いため飛ばします)')
        return
    from run_batch import derive_names
    names = derive_names(['/d/S1/W_antenna1.mat', '/d/S3/J_antenna1.mat', '/d/S2/S2_W_antenna1.mat'],
                         prefix_parent=True)
    check(names == ['S1_W_antenna1', 'S3_J_antenna1', 'S2_W_antenna1'],
          f'セッション名を 1 回だけ前に付ける ({names})')


def test_runner(root):
    print('\n--- run_har.py (doppler_traces 直下に置いた場合) ---')
    flat = os.path.join(root, 'flat')
    work = os.path.join(root, 'work')
    make_traces(flat, flat=True)
    out = run('run_har.py', '--train', 'S1', '--test', 'S2,S3', '--doppler_dir', flat,
              '--work_dir', work, '--epochs', '8', cwd=root)
    check('活動ラベル (ファイル名から): E,J,R,W' in out, '活動ラベルをファイル名から集める')
    summaries = [f for f in os.listdir(os.path.join(work, 'outputs')) if f.startswith('har_summary_')]
    check(len(summaries) == 1, '結果のまとめが outputs/ に出る')
    with open(os.path.join(work, 'outputs', summaries[0]), encoding='utf-8') as fp:
        summary = fp.read()
    for ts in ['S1 の取り分け', 'S2,S3', '| S2 |', '| S3 |']:
        check(ts in summary, f'まとめに「{ts}」の行がある')
    m = load(os.path.join(work, 'outputs', 'complete_different_antenna1,antenna2_E,J,R,W_S2,S3.txt'))
    check(m['accuracy_max_merge'] > 0.9, f'S2,S3 での正解率 (統合) {m["accuracy_max_merge"]:.3f}')
    check(os.path.isdir(os.path.join(flat, 'S1', 'train_antenna1,antenna2_E,J,R,W')),
          'データセットはセッションのサブディレクトリに書く')

    r = subprocess.run([sys.executable, os.path.join(_CODE, 'run_har.py'), '--train', 'S9',
                        '--test', 'S2', '--doppler_dir', flat, '--work_dir', work],
                       cwd=root, capture_output=True, text=True)
    check(r.returncode != 0 and 'S9_ で始まるドップラーデータがありません' in r.stdout,
          '無いセッションを指定すると止まる')


def main():
    workdir = tempfile.mkdtemp(prefix='sharp_usrp_har_test_')
    try:
        make_traces(workdir)
        test_dataset(workdir)
        test_batch_names()
        if importlib.util.find_spec('tensorflow') and importlib.util.find_spec('sklearn'):
            test_network(workdir)
            test_runner(workdir)
        else:
            print('\n(TensorFlow / scikit-learn が無いため学習のテストは飛ばします)')
    except TestFailure as e:
        print(f'\n失敗: {e}')
        return 1
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    print('\nすべてのテストに合格しました。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
