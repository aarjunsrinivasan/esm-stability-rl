"""The train/val carve in train_dpo.load_pairs.

The guarantee under test is group-disjointness: val must be carved by WT domain,
never by row. Carving by row would put mutants of the same wildtype on both sides,
and val reward_acc would then be measuring memorization.
"""
from __future__ import annotations

import pandas as pd

from align import train_dpo


def _pairs_frame(n_wt=10, per_wt=6):
    """Rows are interleaved across wildtypes, not blocked by wildtype. A frame
    blocked by WT would let a naive row-slice split cleanly on a domain boundary
    by luck, and the disjointness assertions below would pass on broken code."""
    return pd.DataFrame([
        {"WT_name": f"wt_{w}", "chosen": f"C{w}{i}", "rejected": f"R{w}{i}",
         "dG_chosen": 2.0, "dG_rejected": 0.5}
        for i in range(per_wt) for w in range(n_wt)
    ])


def _write_pairs_csv(tmp_path, monkeypatch, df):
    path = tmp_path / "dpo_pairs.csv"
    df.to_csv(path, index=False)
    monkeypatch.setattr(train_dpo, "PAIRS_CSV", path)
    return path


def test_runtime_carve_never_lets_a_wildtype_span_train_and_val(tmp_path, monkeypatch):
    _write_pairs_csv(tmp_path, monkeypatch, _pairs_frame())

    train, val = train_dpo.load_pairs(val_frac=0.2, seed=0, max_pairs=None)

    assert not set(train.WT_name) & set(val.WT_name)
    assert len(train) + len(val) == 60          # every pair is accounted for
    assert len(val) > 0 and len(train) > 0


def test_runtime_carve_is_seed_deterministic_and_seed_sensitive(tmp_path, monkeypatch):
    _write_pairs_csv(tmp_path, monkeypatch, _pairs_frame(n_wt=20))

    a_train, a_val = train_dpo.load_pairs(val_frac=0.2, seed=3, max_pairs=None)
    b_train, b_val = train_dpo.load_pairs(val_frac=0.2, seed=3, max_pairs=None)
    c_train, c_val = train_dpo.load_pairs(val_frac=0.2, seed=99, max_pairs=None)

    pd.testing.assert_frame_equal(a_train, b_train)
    pd.testing.assert_frame_equal(a_val, b_val)
    assert set(a_val.WT_name) != set(c_val.WT_name)


def test_runtime_carve_always_yields_at_least_one_val_domain(tmp_path, monkeypatch):
    """n_val = max(1, ...) — a val_frac small enough to round to zero must still
    leave something to evaluate on rather than silently producing an empty val set."""
    _write_pairs_csv(tmp_path, monkeypatch, _pairs_frame(n_wt=4))

    _, val = train_dpo.load_pairs(val_frac=0.01, seed=0, max_pairs=None)

    assert val.WT_name.nunique() == 1


def test_explicit_split_files_are_used_verbatim_and_bypass_the_carve(tmp_path, monkeypatch):
    """build_dpo_pairs.py already guarantees structural disjointness, so load_pairs
    must not re-split what it is handed."""
    _write_pairs_csv(tmp_path, monkeypatch, _pairs_frame(n_wt=50))  # must be ignored

    train_df = _pairs_frame(n_wt=3, per_wt=4)
    val_df = _pairs_frame(n_wt=2, per_wt=4)
    val_df["WT_name"] = "val_" + val_df.WT_name
    train_path, val_path = tmp_path / "train.csv", tmp_path / "val.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    train, val = train_dpo.load_pairs(val_frac=0.5, seed=0, max_pairs=None,
                                      train_path=train_path, val_path=val_path)

    assert len(train) == 12 and len(val) == 8
    assert not set(train.WT_name) & set(val.WT_name)


def test_max_pairs_caps_both_sides_of_an_explicit_split(tmp_path, monkeypatch):
    """Val is re-scored every --eval-steps, so an uncapped val would make eval,
    not training, the bottleneck."""
    train_df, val_df = _pairs_frame(n_wt=10), _pairs_frame(n_wt=8)
    train_path, val_path = tmp_path / "train.csv", tmp_path / "val.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    train, val = train_dpo.load_pairs(val_frac=0.1, seed=0, max_pairs=15,
                                      train_path=train_path, val_path=val_path)

    assert len(train) == 15 and len(val) == 15


def test_max_pairs_larger_than_the_data_keeps_everything(tmp_path, monkeypatch):
    train_df, val_df = _pairs_frame(n_wt=2), _pairs_frame(n_wt=1)
    train_path, val_path = tmp_path / "train.csv", tmp_path / "val.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    train, val = train_dpo.load_pairs(val_frac=0.1, seed=0, max_pairs=10_000,
                                      train_path=train_path, val_path=val_path)

    assert len(train) == len(train_df) and len(val) == len(val_df)
