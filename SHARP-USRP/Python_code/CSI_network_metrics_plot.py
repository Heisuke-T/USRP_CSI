"""
    混同行列の描画

    CSI_network.py / CSI_network_test.py が outputs/ に保存した混同行列を、
    行ごとに正規化 (各正解クラスのうち何割をどう予測したか) して画像にする。
    アンテナ単体と、アンテナ統合後 (_max_merge) の 2 枚を出す。

    例:
      python CSI_network_metrics_plot.py complete_different_antenna1,antenna2_E,W,R_S2,S3 E,W,R
      -> plots/complete_different_..._S2,S3.png, plots/..._max_merge.png

    Copyright (C) 2026 Heisuke Takeda
    Based on SHARP (C) 2022 Francesca Meneghello, GNU GPL v3.
    Released under the GNU GPL v3 (see LICENSE).
"""

import argparse
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_confusion_matrix(conf, activities, out_png, title):
    # 軸ラベルは英語にする。日本語フォントが無い環境で文字化けするため。
    totals = conf.sum(axis=1, keepdims=True)
    norm = np.divide(conf, totals, out=np.zeros(conf.shape, dtype=float), where=totals > 0)
    n = len(activities)
    fig, ax = plt.subplots(figsize=(1.0 * n + 2.5, 1.0 * n + 2))
    im = ax.imshow(norm, vmin=0, vmax=1, cmap='Blues')
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{norm[i, j]:.2f}', ha='center', va='center',
                    color='white' if norm[i, j] > 0.5 else 'black')
    ax.set_xticks(range(n), activities)
    ax.set_yticks(range(n), activities)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('name_file', help='outputs/ 内の結果ファイル名 (.txt を除く)')
    parser.add_argument('activities', help='活動ラベル (カンマ区切り)')
    parser.add_argument('--plot_dir', default='./plots/', help='画像の保存先')
    parser.add_argument('--suffix', default='',
                        help='画像のファイル名の末尾に付ける文字列 (条件ごとに図を分けたいとき)')
    args = parser.parse_args()

    with open(os.path.join('./outputs', args.name_file + '.txt'), 'rb') as fp:
        metrics = pickle.load(fp)
    activities = args.activities.split(',')

    for key, suffix, acc_key in (('conf_matrix', '', 'accuracy_single'),
                                 ('conf_matrix_max_merge', '_max_merge', 'accuracy_max_merge')):
        out_png = os.path.join(args.plot_dir, args.name_file + args.suffix + suffix + '.png')
        plot_confusion_matrix(metrics[key], activities, out_png,
                              f'{args.name_file}{args.suffix}{suffix}\naccuracy {metrics[acc_key]:.3f}')
        print(f'-> {out_png}')


if __name__ == '__main__':
    main()
