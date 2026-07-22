from __future__ import annotations

import pandas as pd

from data import download_swissprot


def _raw_row(entry, ec, seq):
    """One row in UniProt's TSV schema (fields=accession,ec,length,sequence)."""
    return {"Entry": entry, "EC number": ec, "Length": len(seq), "Sequence": seq}


def _raw_frame():
    """Synthetic UniProt TSV covering every filter branch plus one clean protein per
    length bucket. Sequences are poly-A of the right length; only length matters."""
    return pd.DataFrame(
        [
            # clean, one per bucket → all five must survive
            _raw_row("P00001", "1.1.1.1", "A" * 50),     # (0, 75]
            _raw_row("P00002", "2.7.1.1", "A" * 120),    # (75, 150]
            _raw_row("P00003", "3.4.21.4", "A" * 200),   # (150, 250]
            _raw_row("P00004", "4.1.2.13", "A" * 300),   # (250, 350]
            _raw_row("P00005", "5.3.1.9", "A" * 500),    # (350, 512]
            # dropped: two EC numbers → ambiguous label
            _raw_row("P00006", "1.1.1.1; 2.3.1.9", "A" * 200),
            # dropped: partial EC → not a 4-level annotation
            _raw_row("P00007", "3.5.-.-", "A" * 200),
            # dropped: selenocysteine — scored as an ordinary residue by seq_logp
            _raw_row("P00008", "1.8.1.9", "A" * 199 + "U"),
            # dropped: over the 512 cap the query is supposed to enforce
            _raw_row("P00009", "6.1.1.1", "A" * 600),
            # dropped: missing EC
            _raw_row("P00010", None, "A" * 200),
            # dropped: exact duplicate sequence of P00003
            _raw_row("P00011", "3.4.21.4", "A" * 200),
        ]
    )


def test_next_link_survives_the_commas_in_our_own_fields_parameter():
    """Regression: the pagination URL embeds fields=accession,ec,length,sequence, so a
    Link header split on "," tears it into fragments and the crawl dies on page 2."""
    header = (
        '<https://rest.uniprot.org/uniprotkb/search'
        "?fields=accession,ec,length,sequence&query=reviewed%3Atrue"
        '&cursor=abc123&size=500>; rel="next"'
    )

    nxt = download_swissprot._next_link(header)

    assert nxt == (
        "https://rest.uniprot.org/uniprotkb/search"
        "?fields=accession,ec,length,sequence&query=reviewed%3Atrue"
        "&cursor=abc123&size=500"
    )


def test_next_link_returns_none_on_the_last_page():
    assert download_swissprot._next_link(None) is None
    assert download_swissprot._next_link('<https://example.org/x>; rel="prev"') is None


def test_prepare_keeps_one_clean_protein_per_length_bucket():
    out = download_swissprot.prepare(_raw_frame(), n_per_bucket=0)

    assert list(out.columns) == ["accession", "ec", "aa_seq", "length", "length_bucket"]
    assert list(out.accession) == ["P00001", "P00002", "P00003", "P00004", "P00005"]
    assert list(out.length_bucket) == [
        "(0, 75]", "(75, 150]", "(150, 250]", "(250, 350]", "(350, 512]",
    ]
    assert list(out.length) == [50, 120, 200, 300, 500]


def test_prepare_drops_ambiguous_ec_nonstandard_residues_and_oversized_sequences():
    out = download_swissprot.prepare(_raw_frame(), n_per_bucket=0)
    kept = set(out.accession)

    assert "P00006" not in kept, "multi-EC row must be dropped"
    assert "P00007" not in kept, "partial EC (3.5.-.-) must be dropped"
    assert "P00008" not in kept, "sequence with a non-standard residue (U) must be dropped"
    assert "P00009" not in kept, "sequence longer than 512 must be dropped"
    assert "P00010" not in kept, "row with no EC number must be dropped"
    assert "P00011" not in kept, "duplicate sequence must be dropped"
    assert out.aa_seq.str.fullmatch("[ACDEFGHIKLMNPQRSTVWY]+").all()


def test_prepare_length_comes_from_the_sequence_not_the_length_column():
    """UniProt's Length column and the sequence can disagree; the token count the eval
    actually pays for is the sequence, so that must win."""
    df = pd.DataFrame([_raw_row("P00001", "1.1.1.1", "A" * 100)])
    df.loc[0, "Length"] = 999

    out = download_swissprot.prepare(df, n_per_bucket=0)

    assert list(out.length) == [100]
    assert list(out.length_bucket) == ["(75, 150]"]


def test_prepare_subsample_is_capped_per_bucket_and_deterministic():
    df = pd.DataFrame(
        [_raw_row(f"Q{i:05d}", "1.1.1.1", "A" * (100 + i)) for i in range(20)]
        + [_raw_row(f"R{i:05d}", "2.7.1.1", "A" * (300 + i)) for i in range(20)]
    )

    out = download_swissprot.prepare(df, n_per_bucket=3, seed=7)
    out_again = download_swissprot.prepare(df, n_per_bucket=3, seed=7)

    pd.testing.assert_frame_equal(out, out_again)
    assert out.groupby("length_bucket").size().le(3).all()
    assert set(out.length_bucket) == {"(75, 150]", "(250, 350]"}


def test_prepare_subsample_keeps_a_thin_bucket_whole_rather_than_failing():
    """The (0,75] in-distribution control bucket is thin in real Swiss-Prot — asking
    for more than exist must keep what there is, not raise."""
    df = pd.DataFrame(
        [_raw_row("P00001", "1.1.1.1", "A" * 50)]
        + [_raw_row(f"Q{i:05d}", "2.7.1.1", "A" * (300 + i)) for i in range(10)]
    )

    out = download_swissprot.prepare(df, n_per_bucket=5, seed=0)

    assert (out.length_bucket == "(0, 75]").sum() == 1
    assert (out.length_bucket == "(250, 350]").sum() == 5
