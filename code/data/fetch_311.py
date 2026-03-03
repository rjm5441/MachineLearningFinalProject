"""
Fetch a frozen snapshot of the NYC Open Data 311 dataset (erm2-nwe9) and save it to data/raw/.

Why snapshot?
- The dataset is live and changes over time.
- Fair model comparison requires all models to train/evaluate on the same data.
- This script saves a dated CSV plus metadata (row count, query, SHA256 hash).

Usage:
  python -m code.data.fetch_311 --limit 200000
Optional:
  export SOCRATA_APP_TOKEN="your_token_here"
"""

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import requests

from code.config import RAW_DIR, SOCRATA_RESOURCE

DEFAULT_SELECT = "*"
DEFAULT_WHERE = ""  # e.g., "created_date >= '2024-01-01T00:00:00.000'"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_to_csv(out_csv: Path, limit: int, chunk_size: int, select: str, where: str) -> dict:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    headers = {}
    token = os.getenv("SOCRATA_APP_TOKEN")
    if token:
        headers["X-App-Token"] = token

    params_base = {"$select": select}
    if where.strip():
        params_base["$where"] = where.strip()

    total_rows = 0
    offset = 0

    with out_csv.open("wb") as f_out:
        wrote_header = False

        while total_rows < limit:
            batch = min(chunk_size, limit - total_rows)
            params = dict(params_base)
            params["$limit"] = str(batch)
            params["$offset"] = str(offset)

            resp = requests.get(SOCRATA_RESOURCE, params=params, headers=headers, timeout=60)
            resp.raise_for_status()

            content = resp.content
            if not content.strip():
                break

            # Keep header only for the first chunk
            if wrote_header:
                content = b"\n".join(content.splitlines()[1:]) + b"\n"

            if content.strip():
                f_out.write(content)
                wrote_header = True

            rows_in_chunk = max(0, len(resp.text.splitlines()) - (0 if wrote_header and offset == 0 else 1))
            if rows_in_chunk == 0:
                # Fallback: if server returned only header or nothing, stop
                break

            total_rows += rows_in_chunk
            offset += batch

            # Stop if fewer rows returned than requested (end of dataset)
            if len(resp.text.splitlines()) <= 1 + 1:
                break

    metadata = {
        "fetched_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "resource_url": SOCRATA_RESOURCE,
        "limit": limit,
        "chunk_size": chunk_size,
        "soql": {"$select": select, "$where": where.strip() if where.strip() else None},
        "snapshot_csv": str(out_csv),
    }
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200000, help="Max rows to fetch for the snapshot.")
    parser.add_argument("--chunk-size", type=int, default=50000, help="Rows per request (pagination).")
    parser.add_argument("--select", type=str, default=DEFAULT_SELECT, help="SOQL $select, default '*'")
    parser.add_argument("--where", type=str, default=DEFAULT_WHERE, help="SOQL $where clause (optional).")
    args = parser.parse_args()

    date_tag = datetime.utcnow().date().isoformat()
    out_csv = RAW_DIR / f"311_erm2-nwe9_{date_tag}.csv"
    out_meta = RAW_DIR / f"311_erm2-nwe9_{date_tag}.meta.json"

    meta = fetch_to_csv(out_csv, args.limit, args.chunk_size, args.select, args.where)

    # Fill in integrity + size info
    meta["file_bytes"] = out_csv.stat().st_size if out_csv.exists() else 0
    meta["sha256"] = sha256_file(out_csv) if out_csv.exists() else None

    with out_meta.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved snapshot: {out_csv}")
    print(f"Saved metadata: {out_meta}")


if __name__ == "__main__":
    main()
