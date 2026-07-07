"""Download files from the Tsuboyama 2023 mega-scale stability dataset (Zenodo 7992926).

Auto-discovers files + md5 checksums from the Zenodo REST API, supports selective
download by filename substring, resumes partial downloads, and verifies checksums.

Usage:
  # list files only
  python download_tsuboyama.py --list
  # download the processed dG datasets (recommended for the reward/DPO project)
  python download_tsuboyama.py --match Processed_K50_dG
  # download several
  python download_tsuboyama.py --match Processed_K50_dG AlphaFold_model_PDBs
  # download everything
  python download_tsuboyama.py --all
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import requests

RECORD = "7992926"
API = f"https://zenodo.org/api/records/{RECORD}"
OUT = Path(__file__).resolve().parent / "data"


def get_files() -> list[dict]:
    r = requests.get(API, timeout=60)
    r.raise_for_status()
    return r.json()["files"]


def md5_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(f: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    key = f["key"]
    url = f["links"]["self"]
    size = f["size"]
    want_md5 = f["checksum"].split(":", 1)[1]  # "md5:xxxx"
    dest = OUT / key

    # skip if already complete + verified
    if dest.exists() and dest.stat().st_size == size and md5_of(dest) == want_md5:
        print(f"[skip] {key} already present and verified")
        return

    # resume if partial
    pos = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={pos}-"} if pos and pos < size else {}
    mode = "ab" if headers else "wb"
    if pos and not headers:  # existing file wrong size / complete-but-bad -> restart
        pos, mode = 0, "wb"

    print(f"[get ] {key}  ({size/1e6:.1f} MB){'  resuming' if headers else ''}")
    with requests.get(url, stream=True, headers=headers, timeout=120) as resp:
        resp.raise_for_status()
        done = pos
        with open(dest, mode) as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                done += len(chunk)
                pct = 100 * done / size
                print(f"\r       {done/1e6:8.1f}/{size/1e6:.1f} MB ({pct:5.1f}%)", end="")
    print()

    got = md5_of(dest)
    if got == want_md5:
        print(f"[ok  ] {key} md5 verified")
    else:
        print(f"[FAIL] {key} md5 mismatch: got {got}, want {want_md5}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="list files and exit")
    ap.add_argument("--match", nargs="*", default=[], help="filename substrings to download")
    ap.add_argument("--all", action="store_true", help="download all files")
    args = ap.parse_args()

    files = get_files()
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
        print(f"no files matched {args.match}")
        return
    for f in targets:
        download(f)


if __name__ == "__main__":
    main()
