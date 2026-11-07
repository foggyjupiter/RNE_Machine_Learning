"""rneml command line.

    rneml fetch   --dir data/v1                         # download the simulated dataset
    rneml train   --config configs/blind.yaml [--real-dir PATH] [--feature scalogram]
    rneml evaluate outputs/blind_mel [--real-dir PATH]
    rneml ensemble outputs/a outputs/b ... --name ens [--real-dir PATH]
    rneml repeat --config configs/blind.yaml --seeds 0 1 2   # mean ± std over seeds
    rneml compare outputs/a outputs/b                   # paired test on the same clips
    rneml baselines --real-dir PATH                     # RMS / onset / order params / ridge / GBM
    rneml report outputs/*                              # markdown comparison table
    rneml interpret outputs/blind_mel                   # embedding map, probes, band masking
    rneml suite --v1 data/v1 --bb-fixed data/v1/bb_fixed --multi-rpm data/v1/multi_rpm --real-dir PATH [--cpu]
    rneml suite-report                                  # outputs/REPORT.md
    rneml predict rec1.wav rec2.wav --run outputs/blind_mel [--run outputs/...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


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
    p.add_argument("--sets", nargs="*", help="only these: v1 v3 bb_fixed multi_rpm v1_r2 (default all)")

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
    p.add_argument("--arch", choices=["cnn", "resnet"], help="light CNN or full-size ResNet")
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

    p = sub.add_parser("repeat", help="train the same config with several seeds; mean ± std")
    p.add_argument("--config", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--data-dir")
    p.add_argument("--real-dir")
    p.add_argument("--name")
    p.add_argument("--threads", type=int)
    p.add_argument("--width", type=int)
    p.add_argument("--time-pool", type=int)
    p.add_argument("--epochs", type=int)

    p = sub.add_parser("compare", help="paired comparison of two evaluated runs (same clips)")
    p.add_argument("run_a")
    p.add_argument("run_b")
    p.add_argument("--split", default="test")

    p = sub.add_parser("baselines", help="non-deep baselines on the hand-crafted features")
    p.add_argument("--config", default="configs/blind.yaml")
    p.add_argument("--which", nargs="+", default=None,
                   help="subset of: rms onset order_params ridge_all gbm_physics gbm_all")
    p.add_argument("--data-dir")
    p.add_argument("--eval-data-dir")
    p.add_argument("--real-dir")
    p.add_argument("--prefix", default="base_")

    p = sub.add_parser("gap", help="sim-to-real gap of the hand-crafted features, by N")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--real-dir", required=True)
    p.add_argument("--bb-like", action="store_true", help="compare with BB-like simulated material only")
    p.add_argument("--out", help="write the result as JSON")

    p = sub.add_parser("interpret", help="embedding map, linear probes, band masking")
    p.add_argument("run")
    p.add_argument("--data-dir")

    p = sub.add_parser("suite", help="run the report experiment suite (resumable)")
    p.add_argument("--v1", required=True, help="main dataset folder (metadata.csv)")
    p.add_argument("--v1-r2", help="v1 re-rendered with a per-impact radiation spread")
    p.add_argument("--v1-old", help="the first v1 (before calibration), for the R0 experiments")
    p.add_argument("--data", action="append", default=[], metavar="KEY=DIR",
                   help="further data sets by suite key, e.g. v1_sig=data/v1_sig")
    p.add_argument("--bb-fixed", help="fixed-material companion set folder")
    p.add_argument("--multi-rpm", help="multi-condition dataset folder")
    p.add_argument("--real-dir")
    p.add_argument("--max-priority", type=int, default=5)
    p.add_argument("--only", nargs="*")
    p.add_argument("--part", help="one share of the suite for parallel machines (rneml.suite.PARTS)")
    p.add_argument("--reverse", action="store_true", help="run the list last-first (helper machines)")
    p.add_argument("--budget-hours", type=float)
    p.add_argument("--size", choices=["cpu", "default", "full"], default="default",
                   help="cpu: light CNN width 16, 20 ms frames, 30 epochs; full: ResNet + transformer, 80 epochs")
    p.add_argument("--cpu", action="store_true", help="same as --size cpu")
    p.add_argument("--threads", type=int)

    p = sub.add_parser("suite-report", help="collect finished experiments into outputs/REPORT.md")

    p = sub.add_parser("components", help="which sound components tell the count apart (no deep learning)")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--real-dir", required=True)
    p.add_argument("--out", default="outputs/components")
    p.add_argument("--workers", type=int, default=1)

    p = sub.add_parser("predict", help="N and material of new recordings")
    p.add_argument("files", nargs="+", help="wav files")
    p.add_argument("--run", action="append", required=True,
                   help="trained run folder (repeat for an ensemble)")
    p.add_argument("--json", action="store_true")

    a = ap.parse_args(argv)
    if a.cmd == "fetch":
        from rneml.data import REPO_ID, fetch
        print(fetch(a.dir, a.repo or REPO_ID, audio=not a.no_audio, sets=a.sets))
    elif a.cmd == "train":
        from rneml.train import load_config, train
        cfg = load_config(a.config, data_dir=a.data_dir, real_dir=a.real_dir, epochs=a.epochs,
                          name=a.name, device=a.device, feature=a.feature,
                          real_train_per_N=a.real_train_per_n,
                          real_unlabeled_per_N=a.real_unlabeled_per_n, threads=a.threads,
                          seed=a.seed, width=a.width, time_pool=a.time_pool, arch=a.arch)
        print(json.dumps(train(cfg), indent=1))
    elif a.cmd == "evaluate":
        from pathlib import Path

        from rneml.evaluate import evaluate_run
        from rneml.train import load_config
        cfg = load_config(Path(a.run) / "config.yaml", real_dir=a.real_dir)
        if cfg.model_type == "set":
            from rneml.settrain import evaluate_set
            print(json.dumps(evaluate_set(a.run, cfg), indent=1))
        else:
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
    elif a.cmd == "repeat":
        from rneml.stats import aggregate
        from rneml.train import load_config, train
        base = load_config(a.config, data_dir=a.data_dir, real_dir=a.real_dir, threads=a.threads,
                           width=a.width, time_pool=a.time_pool, epochs=a.epochs)
        name = a.name or base.name
        runs = []
        for sd in a.seeds:
            cfg = load_config(a.config, data_dir=a.data_dir, real_dir=a.real_dir, threads=a.threads,
                              width=a.width, time_pool=a.time_pool, epochs=a.epochs,
                              seed=sd, name=f"{name}_s{sd}")
            train(cfg)
            runs.append(Path(cfg.out_dir) / cfg.name)
        agg = aggregate(runs)
        (Path(base.out_dir) / f"{name}_seeds.json").write_text(json.dumps(agg, indent=1))
        print(json.dumps(agg, indent=1))
    elif a.cmd == "compare":
        from rneml.stats import paired_compare
        print(json.dumps(paired_compare(a.run_a, a.run_b, a.split), indent=1))
    elif a.cmd == "baselines":
        from rneml.baselines import BASELINES, run_baseline
        from rneml.train import load_config
        for kind in a.which or BASELINES:
            cfg = load_config(a.config, data_dir=a.data_dir, real_dir=a.real_dir,
                              eval_data_dir=a.eval_data_dir, name=f"{a.prefix}{kind}")
            run_baseline(cfg, kind)
    elif a.cmd == "gap":
        from rneml.gap import BB_LIKE, gap
        from rneml.train import TrainConfig
        res = gap(TrainConfig(data_dir=a.data_dir, real_dir=a.real_dir),
                  material=BB_LIKE if a.bb_like else None)
        if a.out:
            Path(a.out).write_text(json.dumps(res, indent=1))
    elif a.cmd == "interpret":
        from rneml.interpret import interpret
        interpret(a.run, a.data_dir)
    elif a.cmd == "suite":
        from rneml.suite import report, run_suite
        from rneml.suite import SIZES
        common = dict(SIZES["cpu" if a.cpu else a.size])
        if a.threads:
            common["threads"] = a.threads
        from rneml.suite import PARTS
        only = a.only
        if a.part:   # --part with --only: just those experiments of the part (machines sharing a part)
            only = [n for n in PARTS[a.part] if not a.only or n in a.only]
        run_suite({"v1": a.v1, "v1_old": a.v1_old, "v1_r2": a.v1_r2, "bb_fixed": a.bb_fixed,
                   "multi_rpm": a.multi_rpm, **dict(kv.split("=", 1) for kv in a.data)},
                  a.real_dir,
                  max_priority=a.max_priority, only=only, common=common, budget_hours=a.budget_hours,
                  reverse=a.reverse)
        print(report())
    elif a.cmd == "suite-report":
        from rneml.suite import report
        print(report())
    elif a.cmd == "components":
        from rneml.components import run
        run(a.data_dir, a.real_dir, a.out, workers=a.workers)
    elif a.cmd == "predict":
        from rneml.infer import format_result, predict_files
        res = predict_files(a.run, a.files)
        print(json.dumps(res, indent=1) if a.json else "\n".join(format_result(r) for r in res))


if __name__ == "__main__":
    main()
