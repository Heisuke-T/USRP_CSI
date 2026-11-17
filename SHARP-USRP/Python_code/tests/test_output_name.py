"""
    出力名の組み立て (output_name.py) の検証。

      1. 共通語 (WAX202 / aligned / CSI) と stream_0 が落ちること
      2. 掛けた処理が数字の並びになること。--save_unnormalized の版ごとに
         数字が変わること (通常 134 / nowinnorm 13 / raw 1)
      3. 何も掛けていなければ 0 になること
      4. ストリームが 2 本以上あるときは stream_N を残して衝突を避けること
      5. 短くした名前でも、学習側がラベルとアンテナを読めること
         (dataset_utility.find_recordings がアンテナを束ねられる)

    実行:  python3 tests/test_output_name.py

    Copyright (C) 2026 Heisuke Takeda
    Released under the GNU GPL v3 (see LICENSE).
"""

import os
import sys
import tempfile

_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _CODE)

import output_name                                          # noqa: E402
from dataset_utility import find_recordings, parse_antennas  # noqa: E402

_failed = []


def check(cond, msg):
    print(f'  {"OK" if cond else "失敗"}: {msg}')
    if not cond:
        _failed.append(msg)


class Args:
    def __init__(self, despike=False, remove_static=False):
        self.despike = despike
        self.remove_static = remove_static


STEM = 'T0_A_R_antenna1_WAX202_aligned_CSI_stream_0'


def main():
    print('[共通語と stream_0 を落とす]')
    check(output_name.strip_common(STEM) == 'T0_A_R_antenna1_stream_0',
          'WAX202 / aligned / CSI が落ちる')
    check(output_name.split_stream('T0_A_R_antenna1_stream_0')
          == ('T0_A_R_antenna1', '0'), 'stream_0 を切り離す')
    check(output_name.split_stream('T0_A_R_antenna1')
          == ('T0_A_R_antenna1', None), 'stream が無ければそのまま')

    print('\n[掛けた処理の数字]')
    cases = [
        (Args(), 'normalized', 'T0_A_R_antenna1_34'),
        (Args(despike=True), 'normalized', 'T0_A_R_antenna1_134'),
        (Args(despike=True), 'nowinnorm', 'T0_A_R_antenna1_13'),
        (Args(despike=True), 'raw', 'T0_A_R_antenna1_1'),
        (Args(), 'raw', 'T0_A_R_antenna1_0'),
        (Args(despike=True, remove_static=True), 'normalized',
         'T0_A_R_antenna1_1234'),
        (Args(remove_static=True), 'nowinnorm', 'T0_A_R_antenna1_23'),
    ]
    for args, variant, want in cases:
        got = output_name.build(STEM, output_name.flags_for(args, variant))
        check(got == want, f'{variant:10s} despike={args.despike} '
                           f'remove_static={args.remove_static} -> {got}')

    print('\n[数字は常に昇順で、処理の有無だけで決まる]')
    code = output_name.code(output_name.flags_for(
        Args(despike=True, remove_static=True), 'normalized'))
    check(code == '1234' and list(code) == sorted(code), f'{code} が昇順')

    print('\n[ストリームが 2 本以上なら stream_N を残す]')
    one = ['A1_B_W_antenna1_WAX202_aligned_CSI_stream_0']
    two = ['A1_B_W_antenna1_WAX202_aligned_CSI_stream_0',
           'A1_B_W_antenna1_WAX202_aligned_CSI_stream_1']
    got1 = output_name.build_many(one, output_name.flags_for(Args(), 'normalized'))
    got2 = output_name.build_many(two, output_name.flags_for(Args(), 'normalized'))
    check(list(got1.values()) == ['A1_B_W_antenna1_34'],
          '1 本なら stream を落とす')
    check(sorted(got2.values()) == ['A1_B_W_antenna1_stream_0_34',
                                    'A1_B_W_antenna1_stream_1_34'],
          '2 本なら stream を残して衝突を避ける')
    check(len(set(got2.values())) == 2, '2 本の名前が重ならない')

    print('\n[短くした名前でも学習側が読める]')
    with tempfile.TemporaryDirectory() as d:
        made = []
        for act in ('R', 'W'):
            for ant in ('antenna1', 'antenna2'):
                n = output_name.build(
                    f'T0_A_{act}_{ant}_WAX202_aligned_CSI_stream_0',
                    output_name.flags_for(Args(despike=True), 'normalized'))
                open(os.path.join(d, n + '.txt'), 'wb').close()
                made.append(n)
        # ラベルは名前の 4 文字目 (dataset_utility.LABEL_CHAR_INDEX)
        recs = find_recordings(d, 'T0', ['A'], parse_antennas('antenna1,antenna2'))
        check(len(recs) == 2, f'2 測定として束ねる ({len(recs)})')
        check(all(len(files) == 2 for _, files in recs),
              'アンテナ 2 本が 1 測定にまとまる')
        check(all(os.path.basename(f)[:-4] in made
                  for _, files in recs for f in files), '名前が一致する')

    print()
    if _failed:
        print(f'{len(_failed)} 件が失敗しました:')
        for m in _failed:
            print('  -', m)
        return 1
    print('すべてのテストに合格しました。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
