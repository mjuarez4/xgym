import shutil
from pathlib import Path

files = Path().cwd().rglob("*.npz")

for f in files:
    out = f.parent.parent / f"{f.parent.name}_{f.name}"
    shutil.move(f, out)
