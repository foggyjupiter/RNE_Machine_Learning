"""rneml command line.

    rneml fetch   --dir data/v1                         # download the simulated dataset
    rneml train   --config configs/blind.yaml [--real-dir PATH] [--feature scalogram]
    rneml evaluate outputs/blind_mel [--real-dir PATH]
    rneml ensemble outputs/a outputs/b ... --name ens [--real-dir PATH]
    rneml report outputs/*                              # markdown comparison table
    rneml predict rec1.wav rec2.wav --run outputs/blind_mel [--run outputs/...]
"""

from __future__ import annotations

import argparse
import json


def main(argv=None):
    import sys
    sys.stdout.reconfigure(line_buffering=True)   # progress visible when piped
    ap = argparse.ArgumentParser(prog="rneml", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("fetch", help="download the dataset from Hugging Face")
    p.add_argument("--dir", default="data/v1")
    p.add_argument("--repo", default=None)
    p.add_argument("--no-audio", action="store_true")

    p = sub.add_parser("train", help="train and evaluate")
    p.add_argument("--config", default="configs/blind.yaml")
    p.add_argument("--data-dir")
    p.add_argument("--real-dir")
    p.add_argument("--epochs", type=int)
    p.add_argument("--name")
    p.add_argument("--device")
    p.add_argument("--feature", choices=["mel", "scalogram", "mel22k"])
    p.add_argument("--real-train-per-n", type=int, help="labeled real clips per N for training")
    p.add_argument("--real-unlabeled-per-n", type=int,
                   help="unlabeled real clips per N for domain-adversarial adaptation")
    p.add_argument("--threads", type=int, help="torch CPU threads")
    p.add_argument("--width", type=int, help="CNN width (32 default; 16 for CPU)")
    p.add_argument("--time-pool", type=int, help="input frame pooling (1 default; 2 for CPU)")
    p.add_argument("--seed", type=int)

    p = sub.add_parser("evaluate", help="re-evaluate a trained run")
    p.add_argument("run")
    p.add_argument("--real-dir")

    p = sub.add_parser("ensemble", help="evaluate the average of several trained runs")
    p.add_argument("runs", nargs="+")
    p.add_argument("--name", default="ensemble")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--real-dir")

    p = sub.add_parser("report", help="markdown table comparing evaluated runs")
    p.add_argument("runs", nargs="+")

    p = sub.add_parser("predict", help="N and material of new recordings")
    p.add_argument("files", nargs="+", help="wav files")
    p.add_argument("--run", action="append", required=True,
                   help="trained run folder (repeat for an ensemble)")
    p.add_argument("--json", action="store_true")

    a = ap.parse_args(argv)
    if a.cmd == "fetch":
        from rneml.data import REPO_ID, fetch
        print(fetch(a.dir, a.repo or REPO_ID, audio=not a.no_audio))
    elif a.cmd == "train":
        from rneml.train import load_config, train
        cfg = load_config(a.config, data_dir=a.data_dir, real_dir=a.real_dir, epochs=a.epochs,
                          name=a.name, device=a.device, feature=a.feature,
                          real_train_per_N=a.real_train_per_n,
                          real_unlabeled_per_N=a.real_unlabeled_per_n, threads=a.threads,
                          seed=a.seed, width=a.width, time_pool=a.time_pool)
        print(json.dumps(train(cfg), indent=1))
    elif a.cmd == "evaluate":
        from pathlib import Path

        from rneml.evaluate import evaluate_run
        from rneml.train import load_config
        cfg = load_config(Path(a.run) / "config.yaml", real_dir=a.real_dir)
        print(json.dumps(evaluate_run(a.run, cfg), indent=1))
    elif a.cmd == "ensemble":
        from pathlib import Path

        from rneml.evaluate import evaluate_runs
        from rneml.train import load_config
        cfg = load_config(Path(a.runs[0]) / "config.yaml", real_dir=a.real_dir)
        evaluate_runs(a.runs, cfg, Path(a.out_dir) / a.name)
    elif a.cmd == "report":
        from rneml.report import compare
        print(compare(a.runs))
    elif a.cmd == "predict":
        from rneml.infer import format_result, predict_files
        res = predict_files(a.run, a.files)
        print(json.dumps(res, indent=1) if a.json else "\n".join(format_result(r) for r in res))


if __name__ == "__main__":
    main()
