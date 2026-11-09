"""
    回帰の結果 (outputs/regression_<日時>/) の軽いファイルだけを shared_results/ にコピーし、
    今のブランチにコミットして GitHub にプッシュする (Claude が GitHub から読めるように)。

    コピーするもの (既定):
      log.txt, config.json, metrics.csv, metrics_by_trial.csv, cnn_seeds.csv, predictions.csv,
      comparison.png, cnn_gradcam_*.png, importance*.csv, importance_*.png, static_spectrum_*
    --features で features_*.csv (作った特徴量)、--pred_png で pred_*.png (試行ごとの予測の図) も入れる

    使い方:
      python share_results.py                          # いちばん新しい outputs/regression_* を共有
      python share_results.py outputs/regression_20261105_203539
      python share_results.py --features --pred_png    # 特徴量の表と予測の図も
      python share_results.py --no_push                # コミットだけ (プッシュしない)

    コミットするのは shared_results/<フォルダ名>/ だけ。ほかに変更中のファイルがあっても巻き込まない。

    Copyright (C) 2026 Heisuke Takeda
    Released under the GNU GPL v3 (see LICENSE).
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SHARED = os.path.join(HERE, "shared_results")
BASE_FILES = ["log.txt", "config.json", "metrics.csv", "metrics_by_trial.csv", "cnn_seeds.csv", "predictions.csv",
              "comparison.png", "cnn_gradcam_*.png", "importance*.csv", "importance_*.png",
              "static_spectrum_*.csv", "static_spectrum_*.png"]
MAX_MB = 20.0  # stop if the copy gets this large / これより大きければ止める


def git(*args, check=True):
    r = subprocess.run(["git", *args], cwd=HERE, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} に失敗しました:\n{r.stdout}{r.stderr}")
    return r


def latest_result():
    found = sorted(glob.glob(os.path.join(HERE, "outputs", "regression_*")), key=os.path.getmtime)
    found = [d for d in found if os.path.isdir(d)]
    if not found:
        raise RuntimeError("outputs/ に regression_* のフォルダがありません。先に regress_enmo.py を実行してください")
    return found[-1]


def main(argv=None):
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="回帰の結果の軽いファイルを shared_results/ に入れて GitHub にプッシュする")
    ap.add_argument("result", nargs="?", default=None, help="結果のフォルダ（既定: いちばん新しい outputs/regression_*）")
    ap.add_argument("--features", action="store_true", help="features_*.csv（作った特徴量の表）も入れる")
    ap.add_argument("--pred_png", action="store_true", help="pred_*.png（試行ごとの予測の図）も入れる")
    ap.add_argument("--no_push", action="store_true", help="コミットだけしてプッシュしない")
    ap.add_argument("-m", "--message", default=None, help="コミットのメッセージ（既定: results: <フォルダ名>）")
    a = ap.parse_args(argv)
    try:
        src = os.path.abspath(a.result) if a.result else latest_result()
        if not os.path.isdir(src):
            raise RuntimeError(f"フォルダがありません: {src}")
        name = os.path.basename(os.path.normpath(src))
        patterns = BASE_FILES + (["features_*.csv"] if a.features else []) + (["pred_*.png"] if a.pred_png else [])
        files = sorted({f for p in patterns for f in glob.glob(os.path.join(src, p)) if os.path.isfile(f)})
        if not files:
            raise RuntimeError(f"{src} に共有するファイルがありません")
        mb = sum(os.path.getsize(f) for f in files) / 1e6
        if mb > MAX_MB:
            raise RuntimeError(f"合計 {mb:.1f} MB と大きすぎます（上限 {MAX_MB:g} MB）。--features や --pred_png を外してください")
        dst = os.path.join(SHARED, name)
        os.makedirs(dst, exist_ok=True)
        for f in files:
            shutil.copy2(f, os.path.join(dst, os.path.basename(f)))
        print(f"コピーしました: {len(files)} ファイル、{mb:.1f} MB → {os.path.relpath(dst, HERE)}")
        for f in files:
            print(f"  {os.path.basename(f)}")

        branch = git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        rel = os.path.relpath(dst, HERE)
        git("add", "-f", "--", rel)
        if not git("diff", "--cached", "--quiet", "--", rel, check=False).returncode:
            print("変更がありません（前と同じ結果がもう入っています）")
            return 0
        git("commit", "-m", a.message or f"results: {name}", "--", rel)
        print(f"コミットしました（ブランチ {branch}）: {a.message or f'results: {name}'}")
        if a.no_push:
            print("プッシュはしていません（--no_push）")
            return 0
        r = git("push", "origin", branch, check=False)
        if r.returncode != 0:
            # the branch moved on GitHub: bring it in, then push again / GitHub 側が進んでいたら取り込んでからもう一度
            print("プッシュできなかったので、GitHub の変更を取り込んでからもう一度プッシュします …")
            # --autostash: local changes in progress are put aside during the merge and put back after
            # 作業中の変更は取り込みの間だけ退避して、終わったら元に戻す
            git("pull", "--no-rebase", "--no-edit", "--autostash", "origin", branch)
            git("push", "origin", branch)
        print(f"プッシュしました: origin/{branch}  →  Claude に「{name} を見て」と伝えてください")
        return 0
    except RuntimeError as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
