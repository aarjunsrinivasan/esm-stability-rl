"""Eval metrics: the sign convention and the CI math.

The FireProt sign is the single easiest thing in this repo to "fix" into being
wrong — ddG is lower-is-better while every other score here is higher-is-better,
so a stability-tracking model must come out *positive* against -ddG.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from align.eval_fireprot import _stab_rho
from align.eval_fireprot import aggregate as fireprot_aggregate
from align.eval_pppl import aggregate as pppl_aggregate
from align.eval_pppl import paired_bootstrap, sanity_note

# ── FireProt sign convention ─────────────────────────────────────────────────

def test_a_perfect_stability_tracker_scores_plus_one_not_minus_one():
    """Lower ddG = more stable, so a score that rises as ddG falls is *correct*."""
    ddG = [-2.0, -1.0, 0.0, 1.0, 2.0]          # ascending = decreasingly stable
    perfect_score = [5.0, 4.0, 3.0, 2.0, 1.0]  # pseudo-LL falls as stability falls

    assert _stab_rho(perfect_score, ddG) == pytest.approx(1.0)


def test_a_model_that_ranks_stability_backwards_scores_negative():
    ddG = [-2.0, -1.0, 0.0, 1.0, 2.0]
    assert _stab_rho([1.0, 2.0, 3.0, 4.0, 5.0], ddG) == pytest.approx(-1.0)


def test_stab_rho_drops_nan_rows_and_refuses_degenerate_input():
    assert _stab_rho([5.0, 4.0, 3.0, 2.0, np.nan], [-2, -1, 0, 1, np.nan]) == pytest.approx(1.0)
    assert math.isnan(_stab_rho([1.0, 2.0], [1.0, 2.0]))
    assert math.isnan(_stab_rho([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))


def test_fireprot_aggregate_averages_over_proteins_and_skips_undefined_rho():
    per_protein = pd.DataFrame([
        {"model": "aligned", "scoring": "masked", "spearman": 0.4},
        {"model": "aligned", "scoring": "masked", "spearman": 0.6},
        {"model": "aligned", "scoring": "masked", "spearman": -0.2},
        {"model": "aligned", "scoring": "masked", "spearman": np.nan},
        {"model": "base", "scoring": "masked", "spearman": 0.1},
    ])

    out = fireprot_aggregate(per_protein).set_index("model")

    assert out.loc["aligned", "n_proteins"] == 3          # the NaN protein is excluded
    assert out.loc["aligned", "mean_spearman"] == pytest.approx(0.2666667)
    assert out.loc["aligned", "median_spearman"] == pytest.approx(0.4)
    assert out.loc["aligned", "frac_positive"] == pytest.approx(2 / 3)


# ── paired bootstrap ─────────────────────────────────────────────────────────

def test_paired_bootstrap_is_seed_deterministic_and_brackets_the_mean():
    delta = [0.1, 0.2, 0.15, 0.3, 0.05, 0.22, 0.18]

    mean, lo, hi = paired_bootstrap(delta, n_boot=500, seed=0)
    again = paired_bootstrap(delta, n_boot=500, seed=0)

    assert (mean, lo, hi) == again
    assert mean == pytest.approx(float(np.mean(delta)))
    assert lo < mean < hi


def test_paired_bootstrap_reports_no_ci_below_the_minimum_sample():
    mean, lo, hi = paired_bootstrap([0.1, 0.2, 0.3], n_boot=500, seed=0)

    assert mean == pytest.approx(0.2)      # the point estimate still stands
    assert math.isnan(lo) and math.isnan(hi)


def test_paired_bootstrap_ignores_nan_deltas():
    with_nan = paired_bootstrap([0.1, 0.2, 0.15, 0.3, 0.05, np.nan], n_boot=200, seed=1)
    without = paired_bootstrap([0.1, 0.2, 0.15, 0.3, 0.05], n_boot=200, seed=1)

    assert with_nan == without


def test_paired_bootstrap_on_an_all_nan_input_is_nan_not_a_crash():
    mean, lo, hi = paired_bootstrap([np.nan, np.nan], n_boot=100, seed=0)
    assert math.isnan(mean) and math.isnan(lo) and math.isnan(hi)


# ── pppl aggregation ─────────────────────────────────────────────────────────

def _per_seq(n_per_bucket=8, aligned_offset=0.3):
    """Two length buckets, deliberately built out of accession order so a naive
    groupby would report them wrongly."""
    rows = []
    for bucket, length in [("(150, 250]", 200), ("(0, 75]", 50)]:
        for i in range(n_per_bucket):
            base = 1.0 + 0.01 * i
            rows.append({"accession": f"P{length}{i}", "length": length,
                         "length_bucket": bucket, "model": "base", "log_pppl": base})
            rows.append({"accession": f"P{length}{i}", "length": length,
                         "length_bucket": bucket, "model": "aligned",
                         "log_pppl": base + aligned_offset})
    return pd.DataFrame(rows)


def test_pppl_aggregate_orders_buckets_shortest_first_with_all_last():
    """The trend across buckets *is* the result, so order is not cosmetic."""
    out = pppl_aggregate(_per_seq(), n_boot=200, seed=0)

    assert list(out.length_bucket) == ["(0, 75]", "(150, 250]", "all"]


def test_pppl_aggregate_exponentiates_the_log_space_mean_for_display():
    out = pppl_aggregate(_per_seq(), n_boot=200, seed=0).set_index("length_bucket")
    row = out.loc["all"]

    assert row.base_pppl == pytest.approx(math.exp(row.base_log_pppl))
    assert row.aligned_pppl == pytest.approx(math.exp(row.aligned_log_pppl))
    assert row.n == 16


def test_pppl_aggregate_flags_significance_only_when_the_ci_excludes_zero():
    consistent = pppl_aggregate(_per_seq(aligned_offset=0.3), n_boot=400, seed=0)
    assert consistent.set_index("length_bucket").loc["all", "significant"]
    assert consistent.set_index("length_bucket").loc["all", "delta_log_pppl"] == pytest.approx(0.3)

    # A zero delta on every protein cannot be distinguished from noise.
    null = pppl_aggregate(_per_seq(aligned_offset=0.0), n_boot=400, seed=0)
    assert not null.set_index("length_bucket").loc["all", "significant"]


def test_pppl_aggregate_omits_delta_columns_when_only_the_base_model_was_scored():
    base_only = _per_seq()
    base_only = base_only[base_only.model == "base"]

    out = pppl_aggregate(base_only, n_boot=100, seed=0)

    assert "delta_log_pppl" not in out.columns
    assert "base_pppl" in out.columns


# ── base-model plausibility guard ────────────────────────────────────────────

def test_sanity_note_accepts_a_plausible_base_and_warns_at_or_above_uniform():
    plausible = pd.DataFrame([{"length_bucket": "all", "base_pppl": 3.7}])
    broken = pd.DataFrame([{"length_bucket": "all", "base_pppl": 21.0}])

    assert "plausible" in sanity_note(plausible)
    assert "IMPLAUSIBLE" in sanity_note(broken)
    # 20 = uniform over the standard residues, i.e. a model that learned nothing.
    assert "IMPLAUSIBLE" in sanity_note(pd.DataFrame([{"length_bucket": "all",
                                                       "base_pppl": 20.0}]))


def test_sanity_note_is_empty_when_there_is_no_all_row():
    assert sanity_note(pd.DataFrame([{"length_bucket": "(0, 75]", "base_pppl": 3.0}])) == ""
