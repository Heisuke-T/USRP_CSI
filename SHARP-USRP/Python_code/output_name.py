"""
    ドップラーマップの出力名を組み立てる。

    同じ測定を条件を変えて何度も処理すると、どの図がどの条件のものか
    名前から分からなくなる。掛けた処理を数字の並びにして末尾に入れる。

      1 : 外れ値パケットを前後から補間する            (--despike)
      2 : 窓内の時間平均 (静止経路) を引く             (--remove_static)
      3 : パケットごとに、そのパケットの全サブキャリアの平均振幅で割る
      4 : 窓ごとに最大値で割り、雑音床 10^noise_level より小さい値を床にそろえる

    3 と 4 は既定で掛かる。--save_unnormalized で追加保存する版では外れるので、
    同じ測定から 3 通りの名前が出る:

      通常版         3 と 4 あり    T0_A_R_antenna1_134
      nowinnorm 版   3 のみ         T0_A_R_antenna1_13
      raw 版         どちらも無し    T0_A_R_antenna1_1

    どれも掛かっていなければ 0 とする (T0_A_R_antenna1_0)。

    あわせて、測定の区別に関係しない共通語を名前から落とす。受信機の型番
    (WAX202) や、どのファイルにも付く aligned / CSI がそれで、残しても
    名前が長くなるだけ判別には効かない。ストリームが 1 本しかない測定
    (SISO) では stream_0 も落とす。

      T0_A_R_antenna1_WAX202_aligned_CSI_stream_0  ->  T0_A_R_antenna1_134

    学習側 (dataset_utility.py) はラベルを名前の 4 文字目から、アンテナを
    アンダースコア区切りの語から読む。短くするのは末尾だけなので、どちらも
    そのまま動く。

    Copyright (C) 2026 Heisuke Takeda
    Released under the GNU GPL v3 (see LICENSE).
"""

# (数字, 旗の名前, 説明)
STEPS = (
    (1, 'despike', '外れ値パケットの補間'),
    (2, 'remove_static', '窓内の時間平均 (静止経路) の除去'),
    (3, 'packet_norm', 'パケットごとに全サブキャリアの平均振幅で割る'),
    (4, 'window_norm', '窓ごとに最大値で割り雑音床で切り捨てる'),
)

# 測定の区別に効かない共通語。出力名から落とす
COMMON_TOKENS = ('WAX202', 'aligned', 'CSI')

NONE_CODE = '0'


def flags_for(args, variant='normalized'):
    """実行時の引数と版の名前から、各処理を掛けたかどうかを決める。

    variant は CSI_doppler_computation.VARIANTS の名前か 'normalized'。
    """
    return {
        'despike': bool(getattr(args, 'despike', False)),
        'remove_static': bool(getattr(args, 'remove_static', False)),
        'packet_norm': variant != 'raw',
        'window_norm': variant == 'normalized',
    }


def code(flags):
    """旗から数字の並びを作る。何も掛かっていなければ '0'。"""
    digits = ''.join(str(n) for n, key, _ in STEPS if flags.get(key))
    return digits or NONE_CODE


def legend(flags=None):
    """数字の意味を 1 行で返す。ログに出して後から読めるようにする。"""
    if flags is None:
        return ', '.join(f'{n}={desc}' for n, _, desc in STEPS)
    return ', '.join(f'{n}={desc}' for n, key, desc in STEPS if flags.get(key)) \
        or '前処理なし'


def strip_common(stem, drop=COMMON_TOKENS):
    """共通語を落とす。語はアンダースコア区切りで、完全一致のものだけ。"""
    return '_'.join(t for t in stem.split('_') if t not in drop)


def split_stream(stem):
    """末尾の stream_N を切り離す。無ければ (元の名前, None)。"""
    tokens = stem.split('_')
    if len(tokens) >= 3 and tokens[-2] == 'stream' and tokens[-1].isdigit():
        return '_'.join(tokens[:-2]), tokens[-1]
    return stem, None


def build(stem, suffix, keep_stream=False, drop=COMMON_TOKENS):
    """1 つぶんの出力名。suffix は数字の並び、または旗の dict。"""
    base, stream = split_stream(strip_common(stem, drop))
    if keep_stream and stream is not None:
        base = f'{base}_stream_{stream}'
    if not isinstance(suffix, str):
        suffix = code(suffix)
    return f'{base}_{suffix}'


def build_many(stems, suffix, drop=COMMON_TOKENS):
    """複数まとめて {元の名前: 出力名}。

    stream_0 を落とすと別々の測定が同じ名前になってしまう場合 (ストリームが
    2 本以上ある測定) に限り、全部に stream_N を残す。片方だけ残すと
    並べたときに分かりにくいため、残すなら全部に残す。
    """
    short = {s: build(s, suffix, keep_stream=False, drop=drop) for s in stems}
    if len(set(short.values())) < len(short):
        short = {s: build(s, suffix, keep_stream=True, drop=drop) for s in stems}
    return short
