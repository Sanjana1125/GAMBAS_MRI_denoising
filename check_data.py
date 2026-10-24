# explore_paired.py
from pathlib import Path

base = Path("paired_data")  # whatever folder you unzip to

for item in sorted(base.rglob("*"))[:50]:  # first 50 files
    print(item)
