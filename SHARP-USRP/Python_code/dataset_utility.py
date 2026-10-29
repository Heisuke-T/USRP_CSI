"""
    活動認識の学習用データセットまわりの共通処理

    SHARP の dataset_utility.py を元に、USRP の測定に合わせて次を変えている:

      * 活動ラベルはファイル名の 4 文字目 (S1_E_antenna1_... の 'E')。
        SHARP は S1a_E_stream_0 のような名前で 5 文字目を読んでいた
      * アンテナはファイル名中の antenna1 / antenna2 で見分け、使う本数を
        選べるようにした。SHARP は Nexmon の 4 コアを、ファイル名を並べた順に
        n_tot 個ずつ区切って束ねていた
      * ドップラー計算が横に出力する _meta.json を読み込み対象から外す
      * TensorFlow はデータセット作成には不要なので、使う関数の中でだけ読む

    Copyright (C) 2026 Heisuke Takeda
    Based on SHARP (C) 2022 Francesca Meneghello, GNU GPL v3.
    Released under the GNU GPL v3 (see LICENSE).
"""

import glob
import json
import os
import pickle
import shutil

import numpy as np

# 活動ラベルとして読む文字の位置 (0 始まり)。S1_E_antenna1_... の 'E'
LABEL_CHAR_INDEX = 3


def parse_antennas(text):
    """'antenna1,antenna2' または本数 '2' を、アンテナ名のリストにする。"""
    text = text.strip()
    if text.isdigit():
        return [f'antenna{i}' for i in range(1, int(text) + 1)]
    return [a.strip() for a in text.split(',') if a.strip()]


def set_name(kind, antennas, activities):
    """データセットのファイル・ディレクトリ名。

    使うアンテナを名前に含めるので、アンテナ 2 本と 1 本のデータセットは
    互いに上書きせずに並べて置ける。
    例: train_antenna1,antenna2_E,W,R
    """
    return f'{kind}_{",".join(antennas)}_{activities}'


def _trace_files(d, session):
    # _meta.json, _time.npy やデータセットの出力 (labels_*.txt など) を除く
    return [f for f in sorted(os.listdir(d))
            if f.startswith(session + '_') and f.endswith('.txt')]


def session_dirs(base_dir, session):
    """セッションのドップラーデータの場所と、データセットの書き出し先。

    run_batch.py の使い方によって、ドップラーデータの置き場所は 2 通りある:
      セッションのフォルダをまとめて渡した場合  doppler_traces/S1_W_antenna1_stream_0.txt
      --label S1 を付けた場合                  doppler_traces/S1/S1_W_antenna1_stream_0.txt
    サブディレクトリにあればそちら、無ければ直下を使う。データセットは
    どちらの場合も doppler_traces/S1/ に書く (学習・テストのスクリプトはそこを読む)。

    戻り値: (ドップラーデータのディレクトリ, データセットの書き出し先)
    """
    out_dir = os.path.join(base_dir, session)
    if os.path.isdir(out_dir) and _trace_files(out_dir, session):
        return out_dir, out_dir
    return base_dir, out_dir


def find_recordings(trace_dir, session, activities, antennas):
    """セッションのドップラーデータを、測定ごとにアンテナを揃えて束ねる。

    対象は trace_dir 内の <セッション>_ で始まる .txt。同じ測定のファイルどうしは
    アンテナ名の部分だけが違う (S1_E_antenna1_..._stream_0 と S1_E_antenna2_..._stream_0)。

    戻り値: [(ラベル文字, [アンテナ順のファイルパス]), ...]  名前順
    """
    groups = {}
    for f in _trace_files(trace_dir, session):
        name = f[:-4]
        label = name[LABEL_CHAR_INDEX]
        if label not in activities:
            continue
        tokens = name.split('_')
        ant = next((a for a in antennas if a in tokens), None)
        if ant is None:
            continue        # 使わないアンテナ (antenna1 だけで学習するときの antenna2 など)
        key = '_'.join('*' if t == ant else t for t in tokens)
        groups.setdefault(key, (label, {}))[1][ant] = os.path.join(trace_dir, f)

    recordings = []
    for key in sorted(groups):
        label, by_ant = groups[key]
        missing = [a for a in antennas if a not in by_ant]
        if missing:
            print(f'  警告: {key} は {",".join(missing)} が無いため使いません')
            continue
        recordings.append((label, [by_ant[a] for a in antennas]))
    return recordings


def load_recording(files):
    """1 測定ぶんのドップラーデータを読み、[アンテナ x ビン x 時間] にする。

    alignCSI.m の出力 (2 台が両方受けたパケットだけを同じ順番で並べたもの) から
    作ったドップラーデータなら、長さはアンテナ間で一致する。一致しない場合は
    短いほうに合わせて末尾を切るが、そのときは窓の番号が 2 台で同じ時刻を
    指さない (SHARP は 4 コアが同じパケットを受けるので長さが常に一致していた)。

    戻り値: (配列, 各ファイルの _meta.json の中身のリスト)
    """
    mats, metas = [], []
    for f in files:
        with open(f, 'rb') as fp:
            stft = pickle.load(fp)
        # SHARP と同じ: ドップラービンごとに時間平均を引く
        stft = stft - np.mean(stft, axis=0, keepdims=True)
        mats.append(stft.T)
        meta_file = f[:-4] + '_meta.json'
        meta = {}
        if os.path.exists(meta_file):
            with open(meta_file, encoding='utf-8') as fp:
                meta = json.load(fp)
        metas.append(meta)

    lengths = [m.shape[1] for m in mats]
    ll = min(lengths)
    if max(lengths) != ll:
        print(f'    長さがアンテナ間で違うため {ll} に揃えます '
              f'({", ".join(str(n) for n in lengths)})')
        print('    注意: 2 台のパケットが揃っていません。alignCSI.m の出力から作ると'
              '一致します')
    return np.stack([m[:, :ll] for m in mats]), metas


def check_doppler_axes(metas, tolerance=0.1):
    """ドップラー速度軸の範囲 v_max が揃っているかを確かめる。

    v_max = c / (Tc * fc) / 2 はパケットの取得レートで変わる。ネットワークは
    ビン番号しか見ないので、レートの違う測定を混ぜると同じビンが違う速度を
    指すことになる。
    """
    vs = [m['v_max'] for m in metas if 'v_max' in m]
    if not vs:
        return
    lo, hi = min(vs), max(vs)
    print(f'  ドップラー速度軸: ±{lo:.2f}〜±{hi:.2f} m/s ({len(vs)} ファイル)')
    if hi > lo * (1 + tolerance):
        print('  警告: 速度軸の範囲がファイルによって違います (取得レートの差)。'
              'ドップラー計算の --sub_sampling でレートを揃えることを検討してください。')


def detect_activities(base_dir, sessions):
    """ドップラーデータのファイル名から、含まれている活動ラベルを集める。"""
    labels = set()
    for session in sessions:
        trace_dir, _ = session_dirs(base_dir, session)
        labels.update(f[LABEL_CHAR_INDEX] for f in _trace_files(trace_dir, session))
    return sorted(labels)


def reset_dir(path):
    """出力先を空にする (前回の窓のファイルが残らないように)。"""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def remove_files(pattern):
    for f in glob.glob(pattern):
        os.remove(f)


def write_set(exp_dir, name, windows, labels, num_windows):
    """窓ごとのファイルと、ラベル・ファイル名・窓数の一覧を書き出す (SHARP と同じ形式)。"""
    set_dir = os.path.join(exp_dir, name)
    reset_dir(set_dir)
    files = []
    for ii, wind in enumerate(windows):
        name_file = os.path.join(set_dir, f'{ii}.txt')
        files.append(name_file)
        with open(name_file, 'wb') as fp:
            pickle.dump(wind, fp)
    for prefix, obj in (('labels', labels), ('files', files),
                        ('num_windows', np.asarray(num_windows))):
        with open(os.path.join(exp_dir, f'{prefix}_{name}.txt'), 'wb') as fp:
            pickle.dump(obj, fp)
    print(f'  {name}: {len(windows)} 窓')
    if not windows:
        print('  警告: 窓が 1 つも作れません。測定 (ドップラーの時間窓の数) が '
              'windows_length に対して短すぎます。windows_length を小さくするか、長い測定を使ってください')


def read_set(exp_dir, name):
    """write_set で書いたラベルとファイル名の一覧を読む。"""
    with open(os.path.join(exp_dir, f'labels_{name}.txt'), 'rb') as fp:
        labels = pickle.load(fp)
    with open(os.path.join(exp_dir, f'files_{name}.txt'), 'rb') as fp:
        files = pickle.load(fp)
    return labels, files


def convert_to_number(lab, csi_label_dict):
    lab_num = np.argwhere(np.asarray(csi_label_dict) == lab)[0][0]
    return int(lab_num)


def create_windows_antennas(csi_list, labels_list, sample_length, stride_length, remove_mean=False):
    """[アンテナ x ビン x 時間] を時間方向に切って窓にする (SHARP と同じ)。

    保存容量を半分にするため float32 にする。学習時はどのみち float32 に
    変換されるので結果は変わらない。
    """
    csi_matrix_stride = []
    labels_stride = []
    num_windows = []
    for i in range(len(labels_list)):
        csi_i = csi_list[i]
        label_i = labels_list[i]
        len_csi = csi_i.shape[2]
        starts = range(0, len_csi - sample_length, stride_length)
        for ii in starts:
            csi_wind = csi_i[:, :, ii:ii + sample_length, ...]
            if remove_mean:
                csi_mean = np.mean(csi_wind, axis=2, keepdims=True)
                csi_wind = csi_wind - csi_mean
            csi_matrix_stride.append(csi_wind.astype(np.float32))
            labels_stride.append(label_i)
        num_windows.append(len(starts))
    return csi_matrix_stride, labels_stride, num_windows


def expand_antennas(file_names, labels, num_antennas):
    file_names_expanded = [item for item in file_names for _ in range(num_antennas)]
    labels_expanded = [item for item in labels for _ in range(num_antennas)]
    stream_ant = np.tile(np.arange(num_antennas), len(labels))
    return file_names_expanded, labels_expanded, stream_ant


def load_data_single(csi_file_t, stream_a):
    csi_file = csi_file_t
    if isinstance(csi_file_t, (bytes, bytearray)):
        csi_file = csi_file.decode()
    with open(csi_file, "rb") as fp:  # Unpickling
        matrix_csi = pickle.load(fp)
    matrix_csi_single = matrix_csi[stream_a, ...].T
    if len(matrix_csi_single.shape) < 3:
        matrix_csi_single = np.expand_dims(matrix_csi_single, axis=-1)
    return matrix_csi_single.astype(np.float32)


def create_dataset_single(csi_matrix_files, labels_stride, stream_ant, input_shape, batch_size, shuffle, cache_file,
                          prefetch=True, repeat=True):
    import tensorflow as tf

    stream_ant = list(stream_ant)
    dataset_csi = tf.data.Dataset.from_tensor_slices((csi_matrix_files, labels_stride, stream_ant))
    py_funct = lambda csi_file, label, stream: (tf.ensure_shape(tf.numpy_function(load_data_single,
                                                                                  [csi_file, stream],
                                                                                  tf.float32), input_shape), label)
    dataset_csi = dataset_csi.map(py_funct)
    dataset_csi = dataset_csi.cache(cache_file)
    if shuffle:
        dataset_csi = dataset_csi.shuffle(len(labels_stride))
    if repeat:
        dataset_csi = dataset_csi.repeat()
    dataset_csi = dataset_csi.batch(batch_size=batch_size)
    if prefetch:
        dataset_csi = dataset_csi.prefetch(buffer_size=1)
    return dataset_csi


def check_window_shape(file_name, num_antennas, feature_length, sample_length):
    """データセットの窓の形が、学習の引数と合っているかを先に確かめる。

    合っていないと TensorFlow の奥で分かりにくいエラーになるため。
    """
    with open(file_name, 'rb') as fp:
        shape = pickle.load(fp).shape
    expected = (num_antennas, feature_length, sample_length)
    if tuple(shape[:3]) != expected:
        raise ValueError(
            f'{file_name} の形が {tuple(shape)} で、引数から想定した '
            f'(アンテナ数, feature_length, sample_length) = {expected} と合いません。'
            f'データセット作成時のアンテナ・窓長と、ドップラー計算の --n_fft を確認してください。')
