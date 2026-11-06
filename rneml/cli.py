"""rneml command line.

    rneml fetch   --dir data/v1                         # download the simulated dataset
    rneml train   --config configs/blind.yaml [--real-dir PATH] [--feature scalogram]
    rneml evaluate outputs/blind_v1 [--real-dir PATH]
"""

from __future__ import annotations

import argparse
import json


def main(argv=None):
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
    p.add_argument("--feature", choices=["mel", "scalogram"])
    p.add_argument("--real-train-per-n", type=int, help="adapt with this many real clips per N")

    p = sub.add_parser("evaluate", help="re-evaluate a trained run")
    p.add_argument("run")
    p.add_argument("--real-dir")

    a = ap.parse_args(argv)
    if a.cmd == "fetch":
        from rneml.data import REPO_ID, fetch
        print(fetch(a.dir, a.repo or REPO_ID, audio=not a.no_audio))
    elif a.cmd == "train":
        from rneml.train import load_config, train
        cfg = load_config(a.config, data_dir=a.data_dir, real_dir=a.real_dir, epochs=a.epochs,
                          name=a.name, device=a.device, feature=a.feature,
                          real_train_per_N=a.real_train_per_n)
        print(json.dumps(train(cfg), indent=1))
    elif a.cmd == "evaluate":
        from pathlib import Path

        from rneml.evaluate import evaluate_run
        from rneml.train import load_config
        cfg = load_config(Path(a.run) / "config.yaml", real_dir=a.real_dir)
        print(json.dumps(evaluate_run(a.run, cfg), indent=1))


if __name__ == "__main__":
    main()
