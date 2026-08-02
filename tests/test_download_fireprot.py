"""Reshaping ThermoMPNN's fireprot_HF schema into a reward_table-style eval table.

Two things must survive: the mutant sequence has to carry the substitution the row
describes, and ddG must pass through unflipped (lower = more stable). A sign flip
here would invert every FireProt result while still looking self-consistent.
"""
from __future__ import annotations

import pandas as pd

from data import download_fireprot as dfp

WT = "MAKGVLYVD"      # 9 residues, 0-indexed positions 0..8


def _raw_row(pdb_id, wt_aa, mut_aa, pdb_position, ddG=-1.5, seq=WT):
    return {"pdb_id_corrected": pdb_id, "pdb_sequence": seq,
            "wild_type": wt_aa, "mutation": mut_aa, "pdb_position": pdb_position,
            "ddG": ddG, "position": pdb_position + 1, "pH": 7.0}


def test_prepare_applies_the_substitution_and_reports_one_indexed_positions():
    raw = pd.DataFrame([_raw_row("1abc", "K", "A", 2)])

    out = dfp.prepare(raw)

    assert len(out) == 1
    row = out.iloc[0]
    assert row.wt_seq == WT
    assert row.aa_seq == "MAAGVLYVD"       # position 2 K -> A
    assert row.mut_type == "K3A"           # 1-indexed, Tsuboyama style
    assert row.position == 3
    assert set(out.columns) == {"WT_name", "mut_type", "wt_seq", "aa_seq", "ddG",
                                "position", "wild_type", "mutation", "pH"}


def test_prepare_passes_ddg_through_without_flipping_the_sign():
    raw = pd.DataFrame([_raw_row("1abc", "K", "A", 2, ddG=-1.5),
                        _raw_row("1abc", "G", "W", 3, ddG=2.25)])

    out = dfp.prepare(raw).set_index("mut_type")

    assert out.loc["K3A", "ddG"] == -1.5    # negative = stabilizing, unchanged
    assert out.loc["G4W", "ddG"] == 2.25


def test_prepare_drops_rows_whose_wild_type_disagrees_with_the_sequence():
    """A mismatch means the row and the sequence describe different proteins;
    silently keeping it would score a mutant that doesn't exist."""
    raw = pd.DataFrame([_raw_row("1abc", "K", "A", 2),      # WT[2] == 'K'  -> kept
                        _raw_row("1abc", "W", "A", 2)])     # WT[2] != 'W'  -> dropped

    out = dfp.prepare(raw)

    assert list(out.mut_type) == ["K3A"]


def test_prepare_drops_out_of_range_synonymous_and_nonstandard_mutations():
    raw = pd.DataFrame([
        _raw_row("1abc", "K", "A", 2),        # kept
        _raw_row("1abc", "D", "X", 8),        # non-standard mutant residue
        _raw_row("1abc", "M", "M", 0),        # synonymous
        _raw_row("1abc", "K", "A", 99),       # position past the end
    ])

    out = dfp.prepare(raw)

    assert list(out.mut_type) == ["K3A"]


def test_prepare_drops_rows_missing_ddg_or_sequence():
    raw = pd.DataFrame([
        _raw_row("1abc", "K", "A", 2),
        {**_raw_row("1abc", "G", "W", 3), "ddG": None},
    ])

    out = dfp.prepare(raw)

    assert list(out.mut_type) == ["K3A"]


def test_prepare_keeps_proteins_separate_and_dedupes_repeated_measurements():
    other = "MAKGVLYVDQQ"
    raw = pd.DataFrame([
        _raw_row("1abc", "K", "A", 2),
        _raw_row("1abc", "K", "A", 2),                    # exact duplicate
        _raw_row("2xyz", "K", "A", 2, ddG=0.4, seq=other),
    ])

    out = dfp.prepare(raw)

    assert len(out) == 2
    assert set(out.WT_name) == {"1abc", "2xyz"}
    assert out[out.WT_name == "2xyz"].iloc[0].wt_seq == other


def test_prepare_rejects_a_frame_missing_expected_columns():
    import pytest

    with pytest.raises(AssertionError, match="missing expected columns"):
        dfp.prepare(pd.DataFrame([{"pdb_id_corrected": "1abc"}]))
