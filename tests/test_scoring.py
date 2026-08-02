"""Shared eval plumbing in align/scoring.py.

`apply_config` is the repo's one config mechanism — a silent precedence bug here
would mislabel every run's provenance, since config.json records the resolved args.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pytest

from align.scoring import apply_config, coerce_paths, rho


def _parser():
    p = argparse.ArgumentParser()
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--adapter", type=Path, default=None)
    p.add_argument("--max-proteins", type=int, default=None)
    return p


def test_yaml_supplies_defaults(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("beta: 0.5\nmax_proteins: 3\n")
    parser = _parser()

    apply_config(parser, cfg)
    args = parser.parse_args([])

    assert args.beta == 0.5
    assert args.max_proteins == 3


def test_an_explicit_cli_flag_beats_the_yaml(tmp_path):
    """The documented contract: config sets defaults, CLI overrides."""
    cfg = tmp_path / "c.yaml"
    cfg.write_text("beta: 0.5\n")
    parser = _parser()

    apply_config(parser, cfg)
    args = parser.parse_args(["--beta", "0.9"])

    assert args.beta == 0.9


def test_an_unknown_yaml_key_is_a_hard_error_not_a_silent_typo(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("beta: 0.5\nbteta: 0.9\n")

    with pytest.raises(AssertionError, match="bteta"):
        apply_config(_parser(), cfg)


def test_an_empty_yaml_is_accepted(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("")

    apply_config(_parser(), cfg)   # yaml.safe_load returns None; must not crash


def test_a_missing_config_file_is_reported(tmp_path):
    with pytest.raises(AssertionError, match="missing config"):
        apply_config(_parser(), tmp_path / "nope.yaml")


def test_coerce_paths_converts_yaml_supplied_strings_to_path():
    """Guards dests declared without `type=Path`: argparse converts string defaults
    only when the action has a type, so a YAML value otherwise stays a str."""
    args = argparse.Namespace(out_dir="/tmp/some/dir", adapter=Path("/already/a/path"))

    coerce_paths(args, "out_dir", "adapter")

    assert args.out_dir == Path("/tmp/some/dir")
    assert args.adapter == Path("/already/a/path")   # idempotent


def test_coerce_paths_tolerates_none_and_absent_attributes():
    args = argparse.Namespace(adapter=None)

    coerce_paths(args, "adapter", "never_declared")

    assert args.adapter is None
    assert not hasattr(args, "never_declared")


def test_a_path_typed_flag_survives_the_yaml_round_trip(tmp_path):
    """The end-to-end property the eval scripts rely on: a path in the config file
    reaches args as a Path, not a str."""
    cfg = tmp_path / "c.yaml"
    cfg.write_text("adapter: /tmp/some/adapter\n")
    parser = _parser()

    apply_config(parser, cfg)
    args = coerce_paths(parser.parse_args([]), "adapter")

    assert args.adapter == Path("/tmp/some/adapter")
    assert isinstance(args.adapter, Path)


# ── rho ──────────────────────────────────────────────────────────────────────

def test_rho_is_one_for_a_monotone_relationship():
    assert rho([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert rho([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_rho_ignores_rows_where_either_side_is_missing():
    """FireProt has NaN ddG rows; they must drop out rather than poison the stat."""
    clean = rho([1, 2, 3, 4], [1, 2, 3, 4])
    with_nan = rho([1, 2, 3, 4, 5], [1, 2, 3, 4, float("nan")])

    assert with_nan == pytest.approx(clean)


def test_rho_is_undefined_rather_than_wrong_on_degenerate_input():
    assert math.isnan(rho([1, 2], [1, 2]))                  # fewer than 3 usable rows
    assert math.isnan(rho([1, 1, 1, 1], [1, 2, 3, 4]))      # constant x
    assert math.isnan(rho([1, 2, 3, 4], [7, 7, 7, 7]))      # constant y
