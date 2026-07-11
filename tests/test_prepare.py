from __future__ import annotations

import numpy as np
import pandas as pd

from data import prepare


def test_parse_dg_handles_censored_numeric_and_invalid_values():
    assert prepare.parse_dG("<-1") == -1.0
    assert prepare.parse_dG(">5") == 5.0
    assert prepare.parse_dG("2.25") == 2.25
    assert prepare.parse_dG(3) == 3.0
    assert np.isnan(prepare.parse_dG("not-a-number"))
    assert np.isnan(prepare.parse_dG(None))


def test_sample_preference_pairs_respects_margin_cap_and_seed():
    df = pd.DataFrame(
        {
            "WT_name": ["wt_a", "wt_a", "wt_a", "wt_b", "wt_b", "wt_single"],
            "aa_seq": ["AAA", "AAC", "AAG", "BBB", "BBC", "SSS"],
            "dG": [0.0, 1.4, 3.1, -1.0, 2.0, 9.0],
        }
    )

    pairs = prepare.sample_preference_pairs(df, margin=1.0, max_pairs_per_wt=2, seed=7)
    pairs_again = prepare.sample_preference_pairs(df, margin=1.0, max_pairs_per_wt=2, seed=7)

    pd.testing.assert_frame_equal(pairs, pairs_again)
    assert set(pairs.columns) == {
        "WT_name",
        "chosen",
        "rejected",
        "dG_chosen",
        "dG_rejected",
    }
    assert "wt_single" not in set(pairs.WT_name)
    assert pairs.groupby("WT_name").size().le(2).all()
    assert (pairs.dG_chosen > pairs.dG_rejected).all()
    assert ((pairs.dG_chosen - pairs.dG_rejected) >= 1.0).all()


def test_prepare_tsuboyama_writes_reward_table_and_natural_only_dpo_pairs(tmp_path, monkeypatch):
    raw = tmp_path / "tsuboyama"
    raw.mkdir()
    csv_path = raw / "Tsuboyama2023_Dataset2_Dataset3_20230416.csv"
    pd.DataFrame(
        [
            {
                "name": "nat_low",
                "aa_seq": "AAAA",
                "mut_type": "mut",
                "WT_name": "natural_wt",
                "WT_cluster": "123",
                "dG_ML": "0.0",
                "ddG_ML": "<-1",
                "Stabilizing_mut": False,
            },
            {
                "name": "nat_mid",
                "aa_seq": "AAAC",
                "mut_type": "mut",
                "WT_name": "natural_wt",
                "WT_cluster": "123",
                "dG_ML": "2.0",
                "ddG_ML": "0.5",
                "Stabilizing_mut": True,
            },
            {
                "name": "nat_high",
                "aa_seq": "AAAG",
                "mut_type": "wt",
                "WT_name": "natural_wt",
                "WT_cluster": "123",
                "dG_ML": ">5",
                "ddG_ML": ">5",
                "Stabilizing_mut": True,
            },
            {
                "name": "denovo_low",
                "aa_seq": "DDDD",
                "mut_type": "mut",
                "WT_name": "denovo_wt",
                "WT_cluster": "de_novo_cluster",
                "dG_ML": "0.0",
                "ddG_ML": "bad",
                "Stabilizing_mut": False,
            },
            {
                "name": "denovo_high",
                "aa_seq": "DDDE",
                "mut_type": "mut",
                "WT_name": "denovo_wt",
                "WT_cluster": "de_novo_cluster",
                "dG_ML": "4.0",
                "ddG_ML": "1.0",
                "Stabilizing_mut": True,
            },
            {
                "name": "missing_dg",
                "aa_seq": "XXXX",
                "mut_type": "mut",
                "WT_name": "natural_wt",
                "WT_cluster": "123",
                "dG_ML": "bad",
                "ddG_ML": "0.0",
                "Stabilizing_mut": False,
            },
        ]
    ).to_csv(csv_path, index=False)

    monkeypatch.setattr(prepare, "DATA_DIR", tmp_path)
    prepare.prepare_tsuboyama()

    reward = pd.read_csv(tmp_path / "prepared" / "reward_table.csv")
    dpo = pd.read_csv(tmp_path / "prepared" / "dpo_pairs.csv")

    assert list(reward.columns) == [
        "name",
        "WT_name",
        "origin",
        "WT_cluster",
        "mut_type",
        "is_wt",
        "aa_seq",
        "dG",
        "ddG",
    ]
    assert "missing_dg" not in set(reward.name)
    assert reward.set_index("name").loc["nat_high", "dG"] == 5.0
    assert reward.set_index("name").loc["nat_low", "ddG"] == -1.0
    assert np.isnan(reward.set_index("name").loc["denovo_low", "ddG"])
    assert set(reward.loc[reward.WT_cluster.eq("123"), "origin"]) == {"natural"}
    assert set(reward.loc[reward.WT_cluster.eq("de_novo_cluster"), "origin"]) == {"de_novo"}
    assert reward.set_index("name").loc["nat_high", "is_wt"]

    assert not dpo.empty
    assert set(dpo.WT_name) == {"natural_wt"}
    assert set(dpo.chosen).isdisjoint({"DDDD", "DDDE"})
    assert set(dpo.rejected).isdisjoint({"DDDD", "DDDE"})
    assert (dpo.dG_chosen > dpo.dG_rejected).all()
    assert ((dpo.dG_chosen - dpo.dG_rejected) >= 1.0).all()
