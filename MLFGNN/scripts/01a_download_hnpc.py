"""Stage 0: download Head-Neck-PET-CT from the NIH Cancer Research Data Commons.

The collection is no longer served by TCIA's public NBIA API (it answers
"not in public domain") and is absent from the Imaging Data Commons. It now
lives in the CRDC under dbGaP accession phs004225, behind a login. Downloading
therefore needs a Gen3 API key:

    1. Sign in at https://nci-crdc.datacommons.io  (NIH RAS: Login.gov / eRA)
    2. Profile -> "Create API key" -> save the JSON as  credentials.json
    3. python scripts/01a_download_hnpc.py --credentials credentials.json

The GC manifest on the collection page lists every series with its CRDC file
ID; the public index endpoint gives size and MD5, which are verified after
every download. Runs are resumable: a series with a ``.done`` marker is
skipped. Only CT, PT and RTSTRUCT are fetched by default (17 GB of the 24 GB
collection); REG objects can be added for patients whose contours sit on the
planning CT (see ``mlfgnn.data.tcia``).

    --dry-run   prints the per-centre / per-modality totals and exits
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger("download")

COMMONS = "https://nci-crdc.datacommons.io"
UNITS = {"B": 1, "KB": 1e3, "MB": 1e6, "GB": 1e9}


class Gen3Session:
    """Minimal Gen3 client: API key -> access token -> presigned URLs."""

    def __init__(self, credentials: Path, commons: str = COMMONS) -> None:
        self.commons = commons.rstrip("/")
        self.api_key = json.loads(Path(credentials).read_text(encoding="utf-8"))["api_key"]
        self._token: str | None = None
        self._token_time = 0.0

    def token(self) -> str:
        if self._token and time.time() - self._token_time < 1500:   # tokens last ~30 min
            return self._token
        req = urllib.request.Request(
            f"{self.commons}/user/credentials/api/access_token",
            data=json.dumps({"api_key": self.api_key}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            self._token = json.load(resp)["access_token"]
        self._token_time = time.time()
        return self._token

    def presigned_url(self, guid: str) -> str:
        req = urllib.request.Request(
            f"{self.commons}/user/data/download/{guid}",
            headers={"Authorization": f"Bearer {self.token()}"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.load(resp)["url"]

    def index_record(self, guid: str) -> dict:
        """Public metadata (size, md5) -- no auth required."""
        with urllib.request.urlopen(f"{self.commons}/index/{guid}", timeout=60) as resp:
            return json.load(resp)


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path: Path, modalities: list[str]) -> pd.DataFrame:
    m = pd.read_csv(path)
    m = m.rename(
        columns={
            "File ID": "guid", "File Name": "filename", "Participant ID": "patient_id",
            "Image Modality": "modality", "File Size": "size_str",
        }
    )
    m["bytes"] = m["size_str"].map(lambda s: float(s.split()[0]) * UNITS[s.split()[1]])
    m["series_uid"] = m["filename"].str.replace(".zip", "", regex=False)
    m["centre"] = m["patient_id"].str.split("-").str[1]
    return m[m["modality"].isin(modalities)].reset_index(drop=True)


def fetch_series(row, session: Gen3Session, out_root: Path, keep_zip: bool) -> tuple[str, str]:
    target = out_root / row.patient_id / row.modality / row.series_uid
    marker = target / ".done"
    if marker.exists():
        return row.series_uid, "skipped"

    target.mkdir(parents=True, exist_ok=True)
    zip_path = target.with_suffix(".zip")
    part = zip_path.with_suffix(".zip.part")

    record = session.index_record(row.guid)
    expected_md5 = record.get("hashes", {}).get("md5")
    expected_size = int(record.get("size", 0))

    for attempt in range(1, 4):
        try:
            url = session.presigned_url(row.guid)
            with urllib.request.urlopen(url, timeout=600) as resp, part.open("wb") as fh:
                for chunk in iter(lambda: resp.read(1 << 20), b""):
                    fh.write(chunk)
            if expected_size and part.stat().st_size != expected_size:
                raise OSError(f"size mismatch {part.stat().st_size} != {expected_size}")
            if expected_md5 and _md5(part) != expected_md5:
                raise OSError("md5 mismatch")
            part.rename(zip_path)
            break
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            logger.warning("%s attempt %d failed: %s", row.series_uid[-12:], attempt, exc)
            part.unlink(missing_ok=True)
            time.sleep(5 * attempt)
    else:
        return row.series_uid, "failed"

    with zipfile.ZipFile(zip_path) as z:
        z.extractall(target)
    if not keep_zip:
        zip_path.unlink()
    marker.write_text(json.dumps({"guid": row.guid, "md5": expected_md5, "size": expected_size}))
    return row.series_uid, "ok"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default="data/raw/GC_manifest_Head-Neck-PET-CT_20260326.csv")
    p.add_argument("--credentials", default="credentials.json", help="Gen3 API key JSON")
    p.add_argument("--out", default="data/raw/Head-Neck-PET-CT")
    p.add_argument("--modalities", nargs="*", default=["CT", "PT", "RTSTRUCT"])
    p.add_argument("--patients", nargs="*", default=None, help="restrict to these IDs")
    p.add_argument("--centres", nargs="*", default=None, help="restrict to these centres")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--keep-zip", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")

    m = load_manifest(Path(args.manifest), args.modalities)
    if args.patients:
        m = m[m["patient_id"].isin(args.patients)]
    if args.centres:
        m = m[m["centre"].isin(args.centres)]

    summary = m.groupby(["centre", "modality"]).agg(series=("guid", "size"), GB=("bytes", lambda b: b.sum() / 1e9)).round(2)
    print(summary.to_string())
    print(f"\n{m['patient_id'].nunique()} patients, {len(m)} series, {m['bytes'].sum() / 1e9:.1f} GB -> {args.out}")
    if args.dry_run:
        return

    cred = Path(args.credentials)
    if not cred.exists():
        raise SystemExit(
            f"\n{cred} not found.\n"
            "Head-Neck-PET-CT is served by the NIH Cancer Research Data Commons behind a login.\n"
            "  1. sign in at https://nci-crdc.datacommons.io (Login.gov or eRA Commons)\n"
            "  2. Profile -> Create API key -> save as credentials.json\n"
            "  3. re-run this script\n"
            "If the portal reports you lack access to phs004225, request it through dbGaP "
            "(https://dbgap.ncbi.nlm.nih.gov, study phs004225) -- the collection's licence is CC BY 3.0."
        )

    session = Gen3Session(cred)
    try:
        session.token()
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"could not obtain an access token ({exc}); is credentials.json current?") from exc

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    counts = {"ok": 0, "skipped": 0, "failed": 0}
    failed: list[str] = []
    started = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_series, row, session, out_root, args.keep_zip): row for row in m.itertuples()}
        for i, fut in enumerate(as_completed(futures), 1):
            uid, status = fut.result()
            counts[status] += 1
            if status == "failed":
                failed.append(uid)
            if i % 25 == 0 or i == len(futures):
                rate = (time.time() - started) / i
                logger.info("%d/%d  ok=%d skipped=%d failed=%d  eta %.0f min",
                            i, len(futures), counts["ok"], counts["skipped"], counts["failed"],
                            rate * (len(futures) - i) / 60)

    if failed:
        (out_root / "FAILED_SERIES.txt").write_text("\n".join(failed))
        logger.error("%d series failed -- listed in FAILED_SERIES.txt; re-run to retry", len(failed))
    print(f"\ndone: {counts}")


if __name__ == "__main__":
    main()
