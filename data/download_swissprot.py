"""
download_swissprot.py
─────────────────────────────────────────────────────────────────────────────
Build a held-out **natural protein** eval set from Swiss-Prot (UniProtKB
reviewed), for the catastrophic-forgetting check in align/eval_pppl.py.

WHY THIS FILE / PROVENANCE
──────────────────────────
Every other eval in this repo scores ≤75-residue Tsuboyama domains (or ≤448
FireProt mutants), i.e. the same short, stability-flavoured distribution DPO
trained on. None of them can answer "is this still a general protein language
model?" — the KL term in train_dpo.py only measures drift *on the training
distribution*. This set is deliberately the opposite: natural, full-length
(median ~300 residues), functionally diverse proteins that appear in no part of
the reward.

  Source     UniProtKB REST /search (cursor-paginated)   https://rest.uniprot.org/
  Query      reviewed:true AND ec:* AND length:[1 TO 512]
  Release    2026_02  (X-UniProt-Release at pin time; 222,178 total results)

We page through /search at 500 rows/request rather than using the /stream
endpoint, which is the obvious choice and the wrong one: measured, /stream
delivers this query at ~4 KB/s (~3 h for the full result set) while /search
returns 500 rows *with sequences* in ~2 s (~15 min total). Both give identical
data.

UniProt has no commit SHA to pin, so the pin is **release + query + row count**
(the constants below). Unlike download_fireprot.py's git pin, this cannot make
the download reproducible — UniProt has no way to request an old release over
REST. So the script instead *detects* drift: it compares the live
X-UniProt-Release header and result count against the pins and warns loudly if
either moved, and records what it actually got in a .meta.json sidecar next to
the CSV, so any eval run can state which snapshot it scored. To re-pin, bump
UNIPROT_RELEASE / EXPECTED_ROWS below to the reported live values.

Length ≤512 mirrors the ESM3 paper's EC-CATH construction (App. A.1.4.3,
docs/esm3.txt) and keeps the O(L²) masked-pseudo-LL eval tractable. `ec:*`
restricts to enzymes, which is not needed for perplexity but *is* needed for the
EC probe this table is also shaped to feed — and it costs nothing here, since a
base-vs-aligned delta is unaffected by the population being enzymes.

WHAT IT PRODUCES
────────────────
data/prepared/swissprot_eval.csv — one row per protein:
  accession       UniProt accession (P00350, …)
  ec              full 4-level EC number, exactly one, fully specified
  aa_seq          the sequence (standard 20 residues only)
  length          len(aa_seq)
  length_bucket   (0,75] · (75,150] · (150,250] · (250,350] · (350,512]

and data/prepared/swissprot_eval.meta.json with the release/query/counts actually
fetched.

The `(0,75]` bucket is the **in-distribution control** — it's the length range DPO
actually trained on, so it's the bucket where drift is expected. It is thin (few
reviewed enzymes are that short); that's fine, the headline is the trend across
buckets, and per-bucket n is always reported so a thin bucket is visible.

Rows are dropped, and counted, for: more than one EC number (`1.1.1.1; 2.3.1.9`),
a partial EC (`3.5.-.-`), or any non-standard residue (U/X/B/Z/O). The residue
filter matters here in a way it never did for Tsuboyama/FireProt: seq_logp and
masked_seq_logp treat only cls/eos/pad as special, so a selenocysteine would be
scored as an ordinary residue and quietly pollute the perplexity.

USAGE (from rl_esm/)
────────────────────
    pixi run python data/download_swissprot.py                 # → data/prepared/swissprot_eval.csv
    pixi run python data/download_swissprot.py --list          # check pins (one page, seconds)
    pixi run python data/download_swissprot.py --n-per-bucket 200   # re-subsample (uses cache)

The first run pages through ~450 requests (~15 min) and caches the raw TSV under
data/swissprot/; re-runs with a different --seed / --n-per-bucket hit that cache
and are instant. --no-cache forces a re-download.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import time
from datetime import date
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent
PREP_DIR = DATA_DIR / "prepared"
RAW_DEST = DATA_DIR / "swissprot"

# Pinned UniProt snapshot — bump these to the live values to re-pin (see docstring).
UNIPROT_RELEASE = "2026_02"
EXPECTED_ROWS = 222178
QUERY = "reviewed:true AND ec:* AND length:[1 TO 512]"
FIELDS = "accession,ec,length,sequence"
SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
PAGE_SIZE = 500          # UniProt's maximum for /search

AA = set("ACDEFGHIKLMNPQRSTVWY")

# Right-open on the left, closed on the right: (lo, hi]. First bucket is the
# in-distribution control (DPO trained on ≤75-residue domains).
BUCKET_EDGES = [0, 75, 150, 250, 350, 512]


def _next_link(link_header: str | None) -> str | None:
    """UniProt paginates by putting the next cursor URL in the Link header as
    `<url>; rel="next"`. Absent header = last page.

    Matched with a regex rather than splitting on "," — the URL embeds our own
    `fields=accession,ec,length,sequence`, so comma-splitting tears it apart.
    """
    if not link_header:
        return None
    m = re.search(r'<([^>]+)>\s*;\s*rel="next"', link_header)
    return m.group(1) if m else None


def _get(url, params=None, timeout=120, attempts=3):
    """GET with a small retry — this makes ~450 sequential requests, so a single
    transient 5xx must not throw away a 15-minute download."""
    for i in range(attempts):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if i == attempts - 1:
                raise
            wait = 2 ** i
            print(f"[warn] {type(e).__name__} on attempt {i+1}/{attempts}, "
                  f"retrying in {wait}s …")
            time.sleep(wait)


def fetch_raw(cache: bool = True) -> tuple[pd.DataFrame, dict]:
    """Page through the pinned query → (DataFrame, provenance dict).

    Returns the live release/count alongside the data so the caller can record what
    was actually fetched rather than what we hoped to fetch. The concatenated result
    is cached to data/swissprot/ so re-runs (e.g. to re-subsample with a new seed)
    don't re-download.
    """
    RAW_DEST.mkdir(parents=True, exist_ok=True)
    raw_file = RAW_DEST / "swissprot_enzymes.tsv.gz"
    meta_file = RAW_DEST / "swissprot_enzymes.meta.json"

    if cache and raw_file.exists() and meta_file.exists():
        with gzip.open(raw_file, "rt") as fh:
            df = pd.read_csv(fh, sep="\t")
        print(f"[ok  ] cache hit → {raw_file}  ({len(df)} rows)")
        return df, json.loads(meta_file.read_text())

    print(f"[get ] {SEARCH_URL}  query={QUERY!r}")
    r = _get(SEARCH_URL, params={"query": QUERY, "format": "tsv", "fields": FIELDS,
                                 "size": PAGE_SIZE})
    total = int(r.headers.get("x-total-results", 0))
    live_release = r.headers.get("x-uniprot-release", "unknown")
    print(f"[get ] {total} results, release {live_release} — "
          f"~{total // PAGE_SIZE + 1} pages at {PAGE_SIZE}/page")

    pages, n_rows = [], 0
    while True:
        pages.append(r.text)
        n_rows += max(0, r.text.count("\n") - 1)      # minus the header line
        nxt = _next_link(r.headers.get("link"))
        print(f"\r[get ] {n_rows}/{total} rows", end="", flush=True)
        if not nxt:
            break
        r = _get(nxt)
    print()

    # Every page repeats the TSV header; keep the first page whole, drop line 1 of the rest.
    body = pages[0] + "".join(p.split("\n", 1)[1] for p in pages[1:])
    df = pd.read_csv(StringIO(body), sep="\t")

    with gzip.open(raw_file, "wt") as fh:
        fh.write(body)
    print(f"[ok  ] saved raw → {raw_file}  ({raw_file.stat().st_size/1e6:.1f} MB, "
          f"{len(df)} rows)")
    meta = {
        "query": QUERY,
        "fields": FIELDS,
        "pinned_release": UNIPROT_RELEASE,
        "live_release": live_release,
        "pinned_expected_rows": EXPECTED_ROWS,
        "live_rows": len(df),
        "fetched": date.today().isoformat(),
    }
    if live_release != UNIPROT_RELEASE:
        print(f"[WARN] UniProt release moved: pinned {UNIPROT_RELEASE} → live "
              f"{live_release}. Numbers are not comparable to earlier runs on the "
              f"pinned release. Re-pin UNIPROT_RELEASE if this is intended.")
    if len(df) != EXPECTED_ROWS:
        print(f"[WARN] row count moved: pinned {EXPECTED_ROWS} → live {len(df)} "
              f"({len(df) - EXPECTED_ROWS:+d})")
    meta_file.write_text(json.dumps(meta, indent=2))
    return df, meta


def prepare(df: pd.DataFrame, n_per_bucket: int = 80, seed: int = 0) -> pd.DataFrame:
    """Filter the raw UniProt TSV to clean single-EC standard-residue proteins and
    take a deterministic, length-stratified subsample.

    Pure (no HTTP) so it is unit-testable — see tests/test_download_swissprot.py.
    `n_per_bucket=0` keeps every qualifying row.
    """
    need = ["Entry", "EC number", "Length", "Sequence"]
    missing = [c for c in need if c not in df.columns]
    assert not missing, f"UniProt TSV is missing expected columns: {missing}"

    n_raw = len(df)
    df = df.dropna(subset=["Entry", "EC number", "Sequence"]).copy()
    ec = df["EC number"].astype(str).str.strip()

    multi = ec.str.contains(";")
    partial = ec.str.contains("-")
    nonstd = ~df["Sequence"].astype(str).str.fullmatch(f"[{''.join(sorted(AA))}]+")
    keep = ~multi & ~partial & ~nonstd
    print(f"[prep] {n_raw} raw rows → dropped {int(multi.sum())} multi-EC, "
          f"{int(partial.sum())} partial-EC, {int(nonstd.sum())} with non-standard "
          f"residues (overlapping counts)")

    out = pd.DataFrame({
        "accession": df.loc[keep, "Entry"].astype(str),
        "ec": ec[keep],
        "aa_seq": df.loc[keep, "Sequence"].astype(str),
    })
    out["length"] = out.aa_seq.str.len()
    # Trust the sequence, not the Length column — they can disagree, and every
    # downstream cost model is driven by the actual token count.
    out = out[(out.length > BUCKET_EDGES[0]) & (out.length <= BUCKET_EDGES[-1])]
    out["length_bucket"] = pd.cut(out.length, BUCKET_EDGES).astype(str)
    out = out.drop_duplicates(subset=["aa_seq"]).sort_values("accession")

    print(f"[prep] {len(out)} clean proteins across {out.ec.nunique()} EC numbers")

    if n_per_bucket:
        # Shuffle once, then take the first n of each bucket: deterministic under
        # `seed`, and a bucket with fewer than n members is kept whole rather than
        # raising (the (0,75] control bucket is genuinely thin).
        out = (out.sample(frac=1, random_state=seed)
                  .groupby("length_bucket", observed=True)
                  .head(n_per_bucket)
                  .sort_values("accession"))

    out = out.reset_index(drop=True)
    print(f"[prep] subsampled to {len(out)} proteins "
          f"(n_per_bucket={n_per_bucket or 'all'}, seed={seed}); per bucket:")
    print(out.length_bucket.value_counts().reindex(
        [f"({BUCKET_EDGES[i]}, {BUCKET_EDGES[i+1]}]" for i in range(len(BUCKET_EDGES) - 1)]
    ).to_string())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true",
                    help="fetch ONE page and print the live release/count + raw schema/head, "
                         "then exit — cheap way to check the pins before the full pull")
    ap.add_argument("--no-cache", action="store_true",
                    help="re-download even if data/swissprot/ already has the raw TSV")
    ap.add_argument("--n-per-bucket", type=int, default=80,
                    help="proteins to keep per length bucket (0 = all). The masked-pseudo-LL "
                         "eval costs Σ L_i forward passes per model, so this is the main cost knob")
    ap.add_argument("--seed", type=int, default=0, help="subsample seed")
    ap.add_argument("--out", type=Path, default=PREP_DIR / "swissprot_eval.csv")
    args = ap.parse_args()

    if args.list:
        # One page only — never paginate the whole result set just to show a schema.
        r = _get(SEARCH_URL, params={"query": QUERY, "format": "tsv",
                                     "fields": FIELDS, "size": 5})
        live_release = r.headers.get("x-uniprot-release", "unknown")
        live_rows = int(r.headers.get("x-total-results", 0))
        print(f"\nlive release: {live_release}  (pinned {UNIPROT_RELEASE})"
              f"{'  ← MOVED' if live_release != UNIPROT_RELEASE else '  ✓'}")
        print(f"live rows:    {live_rows}  (pinned {EXPECTED_ROWS})"
              f"{'  ← MOVED' if live_rows != EXPECTED_ROWS else '  ✓'}")
        head = pd.read_csv(StringIO(r.text), sep="\t")
        print("\ncolumns:", list(head.columns))
        print("\nhead:\n", head.to_string(max_colwidth=60))
        return

    raw, meta = fetch_raw(cache=not args.no_cache)
    prepared = prepare(raw, n_per_bucket=args.n_per_bucket, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(args.out, index=False)

    meta.update(n_per_bucket=args.n_per_bucket, seed=args.seed, n_prepared=len(prepared))
    meta_path = args.out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[done] wrote {args.out}")
    print(f"[done] wrote {meta_path}  (release {meta['live_release']})")
    # The raw TSV is deliberately kept (unlike download_fireprot.py, which deletes its
    # 2 MB re-fetchable CSV): it's a ~15-minute, ~450-request download, and it's what
    # makes re-subsampling with a different --seed/--n-per-bucket instant.


if __name__ == "__main__":
    main()
