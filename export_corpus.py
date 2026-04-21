# export_corpus.py — run once; edit DATASET_ID, OUT_DIR, CODE_COL, MAX_ROWS as needed
import os
import re
from pathlib import Path

from datasets import load_dataset

DATASET_ID = "AhmedSSoliman/CodeSearchNet-Python"
SPLIT = "train"
OUT_DIR = Path("./csn_corpus")  # your --dataset-dir
CODE_COL = "code"  # change if your print(ds[0]) shows another field
MAX_ROWS = 200_000  # cap for disk/time; must be >= training_size - len(poisons)

def safe_segment(s: str, max_len: int = 120) -> str:
    s = s.replace("/", "_").replace("..", "_")
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)[:max_len]
    return s or "x"

OUT_DIR.mkdir(parents=True, exist_ok=True)

ds = load_dataset(DATASET_ID, split=SPLIT)
n = min(len(ds), MAX_ROWS)
for i in range(n):
    row = ds[i]
    code = row[CODE_COL]
    if not isinstance(code, str) or not code.strip():
        continue
    # Unique path: avoids collisions across rows
    repo = safe_segment(str(row.get("repo", f"r{i}")))
    path = safe_segment(str(row.get("path", "f.py")))
    func = safe_segment(str(row.get("func_name", "fn")), 80)
    rel = Path(repo) / f"{path}__{func}__{i}.py"
    out_path = OUT_DIR / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(code, encoding="utf-8")

print("wrote under", OUT_DIR.resolve())
