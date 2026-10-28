"""Download ABIDE I PCP phenotypes and ROI time series (resumable).

    .venv\\Scripts\\python.exe scripts\\01_download_abide.py
"""
from __future__ import annotations

import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cxb.data import roi_file_url, select_subjects  # noqa: E402


def fetch(url: str, dest: Path, retries: int = 3) -> str:
    if dest.exists() and dest.stat().st_size > 0:
        return "cached"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
                f.write(r.read())
            tmp.replace(dest)
            return "ok"
        except Exception as e:  # network errors are retried, then reported
            err = e
    return f"failed: {err}"


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "configs" / "data.yaml"))
    a = cfg["abide"]
    raw = ROOT / cfg["paths"]["raw"]
    pheno_path = raw / a["phenotype_file"]
    print("phenotypes:", fetch(f"{a['base_url']}/{a['phenotype_file']}", pheno_path))

    pheno = pd.read_csv(pheno_path)
    subs = select_subjects(pheno, a["quality_checked"])
    print(f"subjects: {len(pheno)} in file, {len(subs)} after file/QC filter")

    jobs = []
    for atlas in a["atlases"]:
        for fid in subs["FILE_ID"]:
            url = roi_file_url(a["base_url"], a["pipeline"], a["strategy"], atlas, fid)
            dest = raw / a["pipeline"] / a["strategy"] / atlas / f"{fid}_{atlas}.1D"
            jobs.append((url, dest))

    counts: dict[str, int] = {}
    failures = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        futs = {pool.submit(fetch, u, d): (u, d) for u, d in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            status = fut.result()
            key = status.split(":")[0]
            counts[key] = counts.get(key, 0) + 1
            if key == "failed":
                failures.append((futs[fut][0], status))
            if i % 200 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} files  {counts}")

    if failures:
        log = raw / "download_failures.txt"
        log.write_text("\n".join(f"{u}\t{s}" for u, s in failures))
        print(f"{len(failures)} failures written to {log}; rerun to retry")
        sys.exit(1)
    print("done")


if __name__ == "__main__":
    main()
