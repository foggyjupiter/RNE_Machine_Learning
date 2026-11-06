"""Send one suite part's results back from a parallel machine (cloud session).

    python scripts/push_results.py --part M1

Small result files of every experiment of the part (metrics, predictions, configs,
history, split, plots, FAILED.txt; for M0 also outputs/components/) are committed with
`git add -f` (outputs/ is ignored) and pushed to the current branch, rebasing on the other
machines' pushes. Model checkpoints go to the Hugging Face dataset repo under
experiments/<name>/model.pt (too big for git). Safe to run repeatedly.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rneml.suite import PARTS  # noqa: E402

SMALL = {"metrics.json", "predictions.jsonl", "config.yaml", "history.json", "split.json",
         "FAILED.txt", "provenance.json", "components.json"}


def git(*args, check=True):
    return subprocess.run(["git", *args], check=check, capture_output=True, text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=sorted(PARTS))
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--hf-repo", default="FoggyJupiter/rne-granular-acoustics")
    ap.add_argument("--no-models", action="store_true")
    ap.add_argument("--via-hf", action="store_true",
                    help="machines without push access: upload everything (results and models) to "
                         "the Hugging Face dataset under experiments/<name>/ instead of git; the "
                         "main machine collects them with scripts/collect_hf_results.py")
    a = ap.parse_args()
    out = Path(a.outputs)
    dirs = [out / n for n in PARTS[a.part]] + ([out / "components"] if a.part == "M0" else [])
    files = [f for d in dirs if d.is_dir() for f in sorted(d.iterdir())
             if f.is_file() and (f.name in SMALL or f.suffix in (".png", ".md"))]
    done = [d.name for d in dirs if (d / "metrics.json").exists()]
    print(f"part {a.part}: {len(done)}/{len(PARTS[a.part])} experiments finished, {len(files)} files")
    if a.via_hf:
        from huggingface_hub import CommitOperationAdd, HfApi
        up = files + ([] if a.no_models else [d / "model.pt" for d in dirs if (d / "model.pt").exists()])
        if up:
            ops = [CommitOperationAdd(path_in_repo=f"experiments/{f.parent.name}/{f.name}", path_or_fileobj=str(f))
                   for f in up]
            for i in range(5):
                try:
                    HfApi().create_commit(a.hf_repo, repo_type="dataset", operations=ops,
                                          commit_message=f"suite part {a.part}: {len(done)} finished")
                    print(f"uploaded {len(ops)} files to {a.hf_repo}/experiments/")
                    break
                except Exception as e:
                    print(f"upload failed (try {i + 1}): {e!r}"[:300])
                    time.sleep(2 ** (i + 2))
        return
    if files:
        git("add", "-f", *map(str, files))
        if git("diff", "--cached", "--quiet", check=False).returncode != 0:
            git("commit", "-m", f"실험 결과 {a.part}: {len(done)}개 완료 ({', '.join(done)})")
        branch = git("branch", "--show-current").stdout.strip()
        for i in range(6):
            git("pull", "--rebase", "origin", branch, check=False)
            r = git("push", "origin", f"HEAD:{branch}", check=False)
            if r.returncode == 0:
                print(f"pushed to {branch}")
                break
            print(f"push failed (try {i + 1}): {r.stderr.strip()[:200]}")
            time.sleep(2 ** (i + 1))
        else:
            sys.exit("could not push the results")
    if not a.no_models:
        from huggingface_hub import CommitOperationAdd, HfApi
        ops = [CommitOperationAdd(path_in_repo=f"experiments/{d.name}/model.pt", path_or_fileobj=str(d / "model.pt"))
               for d in dirs if (d / "model.pt").exists()]
        if ops:
            HfApi().create_commit(a.hf_repo, repo_type="dataset", operations=ops,
                                  commit_message=f"models of suite part {a.part}")
            print(f"uploaded {len(ops)} models to {a.hf_repo}/experiments/")


if __name__ == "__main__":
    main()
