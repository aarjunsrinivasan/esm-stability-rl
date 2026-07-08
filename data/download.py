"""Download a registered dataset from Zenodo.

Auto-discovers files + md5 checksums from the Zenodo REST API, supports selective
download by filename substring, resumes partial downloads, and verifies checksums.
Downloaded files land in data/<dataset>/.

Usage:
  # list available files for a dataset
  python data/download.py --dataset tsuboyama --list
  # download the processed dG tables (recommended for the reward/DPO project)
  python data/download.py --dataset tsuboyama --match Processed_K50_dG
  # download multiple subsets
  python data/download.py --dataset tsuboyama --match Processed_K50_dG AlphaFold_model_PDBs
  # download everything
  python data/download.py --dataset tsuboyama --all
  # arbitrary Zenodo record (bypasses the dataset registry)
  python data/download.py --record 7992926 --dest my_data --all
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parent

# Registry: dataset name → Zenodo record id + local subdirectory name
DATASETS: dict[str, dict] = {
    "tsuboyama": {
        "record": "7992926",
        "dest":   "tsuboyama",
        "description": "Tsuboyama 2023 mega-scale folding stability (Zenodo 7992926)",
    },
}


def get_files(record: str) -> list[dict]:
    url = f"https://zenodo.org/api/records/{record}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()["files"]


def md5_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(f: dict, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    key      = f["key"]
    url      = f["links"]["self"]
    size     = f["size"]
    want_md5 = f["checksum"].split(":", 1)[1]   # "md5:xxxx"
    dest     = dest_dir / key

    if dest.exists() and dest.stat().st_size == size and md5_of(dest) == want_md5:
        print(f"[skip] {key} already present and verified")
        return

    pos     = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={pos}-"} if pos and pos < size else {}
    mode    = "ab" if headers else "wb"
    if pos and not headers:
        pos, mode = 0, "wb"

    print(f"[get ] {key}  ({size/1e6:.1f} MB){'  resuming' if headers else ''}")
    with requests.get(url, stream=True, headers=headers, timeout=120) as resp:
        resp.raise_for_status()
        done = pos
        with open(dest, mode) as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                done += len(chunk)
                print(f"\r       {done/1e6:8.1f}/{size/1e6:.1f} MB ({100*done/size:5.1f}%)", end="")
    print()

    got = md5_of(dest)
    if got == want_md5:
        print(f"[ok  ] {key} md5 verified")
    else:
        print(f"[FAIL] {key} md5 mismatch: got {got}, want {want_md5}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", choices=list(DATASETS), metavar="NAME",
                     help=f"registered dataset ({', '.join(DATASETS)})")
    src.add_argument("--record", metavar="ID",
                     help="arbitrary Zenodo record id (use with --dest)")
    ap.add_argument("--dest", metavar="DIR",
                    help="local subdirectory under data/ (required with --record)")
    ap.add_argument("--list",  action="store_true", help="list files and exit")
    ap.add_argument("--match", nargs="*", default=[], metavar="SUBSTR",
                    help="filename substrings to download (space-separated)")
    ap.add_argument("--all",   action="store_true", help="download all files")
    args = ap.parse_args()

    if args.dataset:
        entry    = DATASETS[args.dataset]
        record   = entry["record"]
        dest_dir = DATA_DIR / entry["dest"]
        print(f"dataset: {entry['description']}")
    else:
        if not args.dest:
            ap.error("--dest DIR is required when using --record")
        record   = args.record
        dest_dir = DATA_DIR / args.dest

    files = get_files(record)

    if args.list or (not args.match and not args.all):
        for f in files:
            print(f"{f['size']/1e6:8.1f} MB  {f['key']}")
        if not args.list:
            print("\npass --match <substr> or --all to download")
        return

    targets = files if args.all else [
        f for f in files if any(m.lower() in f["key"].lower() for m in args.match)
    ]
    if not targets:
        print(f"no files matched {args.match!r}")
        return
    for f in targets:
        download(f, dest_dir)


if __name__ == "__main__":
    main()
