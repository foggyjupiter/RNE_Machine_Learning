"""Merge downloaded data sets into one training folder (no copies: the audio stays where it is).

    python scripts/merge_datasets.py OUT SRC1 SRC2 ...

Writes OUT/metadata.csv with the columns common to all sources; file_name becomes
<source folder name>/<file_name> and OUT/<source folder name> is a symlink to the source.
pair_id / run_id get the source name as prefix so contrast pairs never cross data sets.
Each source keeps its own split column (train / val / test / test_ood / pairs).
"""
import csv
import sys
from pathlib import Path


def main(out, *srcs):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tables = []
    for s in map(Path, srcs):
        with open(s / "metadata.csv", newline="", encoding="utf-8") as fh:
            tables.append((s, list(csv.DictReader(fh))))
        link = out / s.name
        if not link.exists():
            link.symlink_to(s.resolve(), target_is_directory=True)
    cols = [c for c in tables[0][1][0] if all(c in t[0] for _, t in tables)]
    n = 0
    with open(out / "metadata.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for s, rows in tables:
            for r in rows:
                r = {c: r[c] for c in cols}
                r["file_name"] = f"{s.name}/{r['file_name']}"
                for k in ("run_id", "pair_id"):
                    if k in r and r[k]:
                        r[k] = f"{s.name}:{r[k]}"
                w.writerow(r)
                n += 1
    print(f"{out}: {n} clips from {len(srcs)} sets, {len(cols)} columns")


if __name__ == "__main__":
    main(*sys.argv[1:])
