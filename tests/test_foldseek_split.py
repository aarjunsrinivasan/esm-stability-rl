"""Structural split assignment.

Two separate guarantees live here. `structural_split` protects against
near-duplicate structures spanning train and eval. `denovo_safe_split` protects
the *pretraining*-leakage guarantee: de novo domains are held out precisely
because they cannot be in ESM-C's pretraining corpus, so any path that lets one
reach train — or that trains on a natural near-duplicate of one — destroys the
only eval that tests generalization beyond memorization.
"""
from __future__ import annotations

import pandas as pd

from data import foldseek_split as fs


def _clustered_frame():
    """Hand-built clusters covering every branch:

      c_nat_single  1 natural            -> pure-natural singleton
      c_nat_multi   3 naturals           -> pure-natural redundant cluster
      c_mixed       1 natural + 1 de novo -> the contamination case
      c_dn          2 de novo            -> all-de-novo cluster
    """
    rows = [
        ("nat_solo", "c_nat_single", "natural"),
        ("nat_a", "c_nat_multi", "natural"),
        ("nat_b", "c_nat_multi", "natural"),
        ("nat_c", "c_nat_multi", "natural"),
        ("nat_near_dn", "c_mixed", "natural"),
        ("dn_x", "c_mixed", "de_novo"),
        ("dn_y", "c_dn", "de_novo"),
        ("dn_z", "c_dn", "de_novo"),
    ]
    return pd.DataFrame(rows, columns=["WT_name", "foldseek_cluster", "origin"])


# ── denovo-safe assignment ───────────────────────────────────────────────────

def test_every_de_novo_domain_lands_in_test():
    out = fs.denovo_safe_split(_clustered_frame(), seed=0)
    de_novo = out[out.origin == "de_novo"]

    assert set(de_novo.split_representative_denovo_safe) == {"test"}
    assert set(de_novo.split_full_denovo_safe) == {"test"}


def test_a_natural_domain_clustering_with_de_novo_is_excluded_never_trained_on():
    """Training on nat_near_dn would leak the structure of a test example."""
    out = fs.denovo_safe_split(_clustered_frame(), seed=0).set_index("WT_name")

    assert out.loc["nat_near_dn", "split_representative_denovo_safe"] == "excluded"
    assert out.loc["nat_near_dn", "split_full_denovo_safe"] == "excluded"


def test_train_and_val_are_natural_only_and_disjoint_from_test():
    out = fs.denovo_safe_split(_clustered_frame(), seed=0)

    for col in ("split_representative_denovo_safe", "split_full_denovo_safe"):
        trainval = out[out[col].isin(["train", "val"])]
        assert set(trainval.origin) == {"natural"}
        assert not set(trainval.foldseek_cluster) & set(out[out[col] == "test"].foldseek_cluster)


def test_pure_natural_singleton_goes_to_val_and_redundant_cluster_to_train():
    out = fs.denovo_safe_split(_clustered_frame(), seed=0).set_index("WT_name")

    assert out.loc["nat_solo", "split_representative_denovo_safe"] == "val"
    assert out.loc["nat_solo", "split_full_denovo_safe"] == "val"

    multi = ["nat_a", "nat_b", "nat_c"]
    # representative keeps exactly one of the three; full keeps all three.
    assert list(out.loc[multi, "split_representative_denovo_safe"]).count("train") == 1
    assert list(out.loc[multi, "split_full_denovo_safe"]) == ["train"] * 3


# ── origin-agnostic assignment ───────────────────────────────────────────────

def _stub_foldseek(monkeypatch, tmp_path, df):
    """foldseek is an external binary; stubbing the shell-out and the structure-file
    lookup leaves the cluster-to-split assignment — the logic under test — intact."""
    stems = {w: w for w in df.WT_name}
    clusters = dict(zip(df.WT_name, df.foldseek_cluster, strict=True))
    monkeypatch.setattr(fs, "resolve_structure_stems", lambda names: (stems, []))
    monkeypatch.setattr(fs, "run_foldseek", lambda work_dir, cov: tmp_path / "clu.tsv")
    monkeypatch.setattr(fs, "parse_cluster_tsv", lambda p: clusters)
    return df.drop(columns=["foldseek_cluster"])


def test_structural_split_sends_singletons_to_the_eval_pool_and_keeps_one_representative(
        tmp_path, monkeypatch):
    df = _clustered_frame()
    out = fs.structural_split(_stub_foldseek(monkeypatch, tmp_path, df),
                              seed=0).set_index("WT_name")

    assert out.loc["nat_solo", "split_representative"] == "eval_pool"
    assert out.loc["nat_solo", "split_full"] == "eval_pool"

    multi = ["nat_a", "nat_b", "nat_c"]
    assert list(out.loc[multi, "split_representative"]).count("train") == 1
    assert list(out.loc[multi, "split_representative"]).count("excluded") == 2
    assert list(out.loc[multi, "split_full"]) == ["train"] * 3


def test_structural_split_pulls_de_novo_into_train_which_is_why_the_safe_variant_exists(
        tmp_path, monkeypatch):
    """Pins the trap the README warns about: clustering purely by structure has no
    notion of origin, so de novo domains reach train. A future 'simplification'
    collapsing the two variants into one should fail here rather than silently ship."""
    df = _clustered_frame()
    out = fs.structural_split(_stub_foldseek(monkeypatch, tmp_path, df), seed=0)

    trained_on = out[out.split_full == "train"]
    assert "de_novo" in set(trained_on.origin)

    # And the denovo-safe variant of the same frame does not have that property.
    safe = fs.denovo_safe_split(_clustered_frame(), seed=0)
    assert set(safe[safe.split_full_denovo_safe == "train"].origin) == {"natural"}


# ── helpers ──────────────────────────────────────────────────────────────────

def test_parse_cluster_tsv_maps_members_to_representatives(tmp_path):
    tsv = tmp_path / "clu.tsv"
    tsv.write_text("rep1\trep1\nrep1\tmem2\nrep3\trep3\n")

    assert fs.parse_cluster_tsv(tsv) == {"rep1": "rep1", "mem2": "rep1", "rep3": "rep3"}


def test_pppl_bin_edges_are_left_closed_right_open_with_an_overflow_label():
    assert fs.pppl_bin(0.0) == "low"
    assert fs.pppl_bin(1.99) == "low"
    assert fs.pppl_bin(2.0) == "medium"      # boundary belongs to the upper bin
    assert fs.pppl_bin(7.99) == "medium"
    assert fs.pppl_bin(8.0) == "high"
    assert fs.pppl_bin(20.0) == "out_of_range"
