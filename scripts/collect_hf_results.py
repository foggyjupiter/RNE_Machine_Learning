"""Main machine: copy results uploaded by other machines (push_results.py --via-hf) from
the Hugging Face dataset's experiments/ into outputs/ (only experiments not finished here)."""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download

repo = sys.argv[1] if len(sys.argv) > 1 else "FoggyJupiter/rne-granular-acoustics"
out = Path("outputs")
with tempfile.TemporaryDirectory() as tmp:
    snapshot_download(repo, repo_type="dataset", local_dir=tmp, allow_patterns=["experiments/**"],
                      ignore_patterns=["*.pt"])
    src = Path(tmp) / "experiments"
    n = 0
    for d in sorted(src.iterdir()) if src.exists() else []:
        dst = out / d.name
        if (dst / "metrics.json").exists() and not (d / "metrics.json").exists():
            continue
        dst.mkdir(parents=True, exist_ok=True)
        for f in d.iterdir():
            shutil.copy2(f, dst / f.name)
        n += 1
print(f"collected {n} experiment folders from {repo}/experiments/")
