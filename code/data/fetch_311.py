"""
Fetch a frozen snapshot of the NYC Open Data 311 dataset (erm2-nwe9) and save it to data/raw/.
This script saves a dated CSV plus metadata (row count, query, SHA256 hash).

How to run:
  python -m code.data.fetch_311 --limit 200000
"""

import argparse     # For parsing command-line arguments to customize the fetch operation (e.g., limit, chunk size, select/where clauses).
import hashlib      # For computing the SHA256 hash of the downloaded CSV file to ensure data integrity.
import json         # For saving metadata about the fetch operation in a structured format (e.g., resource URL, query parameters, file size).   
import os           # For accessing environment variables (e.g., Socrata API token) and handling file paths.
from datetime import datetime   # For timestamping the fetch operation and naming output files with the current date.
from pathlib import Path        # For convenient and cross-platform handling of file paths when saving the CSV and metadata files.

import time     # For implementing retry logic when making API requests to handle errors.
from requests.exceptions import ReadTimeout, ConnectionError, HTTPError # For catching specific exceptions that may occur during API requests, allowing for retries on timeouts and server errors.

from dateutil import parser # For parsing date strings in the SOQL $where clause, if needed.
import requests     # For making HTTP requests to the Socrata API to fetch the 311 dataset in CSV format.

from code.config import RAW_DIR, SOCRATA_RESOURCE   # Importing configuration constants for the raw data directory and the Socrata API resource URL.

# Default SOQL $select and $where clauses to specify which fields to retrieve and any filtering conditions for the 311 dataset. 
# These can be overridden via command-line arguments.
DEFAULT_SELECT = "unique_key,created_date,closed_date,agency,complaint_type,descriptor,borough,location_type,incident_zip,latitude,longitude,community_board,council_district,police_precinct"
DEFAULT_WHERE = ""  # e.g., "created_date >= '2024-01-01T00:00:00.000'"

# Computes the SHA256 hash of a file at the given path, reading in chunks to handle large files efficiently.
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

# Fetches data from the Socrata API in chunks using pagination, 
# writing directly to a CSV file. 
# Handles retries on timeouts and server errors, 
# and stops when the specified limit is reached or no more data is available. 
# Returns metadata about the fetch operation.
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

    # Open the output CSV file for writing in binary mode to handle the raw content from the API, and write in chunks as they are fetched.
    with out_csv.open("wb") as f_out:
        wrote_header = False # Track whether we've written the header row to avoid duplicating it in subsequent chunks.

        # Loop until we've fetched the desired number of rows (limit) or there are no more rows to fetch.
        while total_rows < limit:
            batch = min(chunk_size, limit - total_rows)
            params = dict(params_base)
            params["$limit"] = str(batch)
            params["$offset"] = str(offset)

            # Implement retry logic for API requests, with exponential backoff on timeouts and server errors.
            timeout = getattr(args, "timeout", 180) if "args" in globals() else 180  # or pass timeout in
            retries = getattr(args, "retries", 5) if "args" in globals() else 5

            last_err = None
            # Try to fetch the data with retries on failure. If a request fails due to a timeout or connection error, it will retry up to the specified number of retries, 
            # If it encounters an HTTP error, it will only retry on server errors (5xx), and will raise the error immediately for client errors (4xx).
            for attempt in range(retries):
                try:
                    resp = requests.get(SOCRATA_RESOURCE, params=params, headers=headers, timeout=timeout)
                    resp.raise_for_status()
                    last_err = None
                    break
                except (ReadTimeout, ConnectionError) as e:
                    last_err = e
                except HTTPError as e:
                    # Retry on server errors; don't retry on 4xx (bad query)
                    status = getattr(e.response, "status_code", None)
                    if status is not None and 500 <= status < 600:
                        last_err = e
                    else:
                        raise

                sleep_s = min(2 ** attempt, 30)
                time.sleep(sleep_s)

            # If we exhausted retries and still have an error, raise it to stop the process.
            if last_err is not None:
                raise last_err

            # If the request was successful, process the content. 
            # If the content is empty or only contains whitespace, it indicates we've reached the end of the dataset, 
            # and we break the loop.
            content = resp.content
            if not content.strip():
                break

            # Keep header only for the first chunk
            if wrote_header:
                content = b"\n".join(content.splitlines()[1:]) + b"\n"

            # Write the chunk to the output CSV file. If the content is not empty, write it 
            # and mark that we've written the header (if it was the first chunk).
            if content.strip():
                f_out.write(content)
                wrote_header = True

            # Count the number of rows in the chunk (excluding the header if it's not the first chunk) 
            # to update our total row count.
            rows_in_chunk = max(0, len(resp.text.splitlines()) - (0 if wrote_header and offset == 0 else 1))
            if rows_in_chunk == 0:
                # Fallback: if server returned only header or nothing, stop
                break

            total_rows += rows_in_chunk
            offset += batch

            # Stop if fewer rows returned than requested (end of dataset)
            if len(resp.text.splitlines()) <= 1 + 1:
                break

    # After fetching is complete, compile metadata about the fetch operation, 
    # including the timestamp, resource URL, query parameters, and the path to the saved CSV file.
    metadata = {
        "fetched_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "resource_url": SOCRATA_RESOURCE,
        "limit": limit,
        "chunk_size": chunk_size,
        "soql": {"$select": select, "$where": where.strip() if where.strip() else None},
        "snapshot_csv": str(out_csv),
    }
    return metadata

# Main function to parse command-line arguments, fetch the 311 dataset snapshot to CSV, 
# compute metadata including file size and SHA256 hash, 
# and save both the data and metadata to the RAW_DIR with a timestamped filename.
def main():
    # Set up command-line argument parsing to allow customization of the fetch operation,
    #  such as the number of rows to fetch (limit), 
    # the chunk size for pagination, 
    # the SOQL select and where clauses, 
    # and retry settings.
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200000, help="Max rows to fetch for the snapshot.")
    parser.add_argument("--chunk-size", type=int, default=50000, help="Rows per request (pagination).")
    parser.add_argument("--select", type=str, default=DEFAULT_SELECT, help="SOQL $select, default '*'")
    parser.add_argument("--where", type=str, default=DEFAULT_WHERE, help="SOQL $where clause (optional).")
    parser.add_argument("--timeout", type=int, default=240, help="Request timeout in seconds.")
    parser.add_argument("--retries", type=int, default=5, help="Number of retries on timeout/5xx.")
    args = parser.parse_args()
    
    # Create output file paths with a timestamp to ensure each snapshot is 
    # uniquely identified by the date it was fetched.
    date_tag = datetime.utcnow().date().isoformat()
    out_csv = RAW_DIR / f"311_erm2-nwe9_{date_tag}.csv"
    out_meta = RAW_DIR / f"311_erm2-nwe9_{date_tag}.meta.json"

    # Fetch the data and save to CSV, while also collecting metadata about the fetch operation.
    meta = fetch_to_csv(out_csv, args.limit, args.chunk_size, args.select, args.where)

    # Fill in integrity + size info
    meta["file_bytes"] = out_csv.stat().st_size if out_csv.exists() else 0
    meta["sha256"] = sha256_file(out_csv) if out_csv.exists() else None

    # Save metadata to a JSON file for record-keeping and future reference about the fetch operation.
    with out_meta.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Print to the console the paths to the saved snapshot and metadata.
    print(f"Saved snapshot: {out_csv}")
    print(f"Saved metadata: {out_meta}")


if __name__ == "__main__":
    main()
