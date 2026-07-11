"""
sweep_dpo.py
─────────────────────────────────────────────────────────────────────────────
Optuna hyperparameter search over align/train_dpo.py: β × lr × lora_rank ×
batch_size. Runs trials in-process (reusing train_dpo.build_args/run directly,
one fresh model per trial) rather than as subprocesses, so Optuna's pruner can
see intermediate reward_acc values (reported from inside train_dpo.run's
run_eval()) and kill obviously-bad trials early.

Search space + fixed args come from a YAML config (align/configs/sweep.yaml):
  fixed:         args passed straight through to train_dpo.py's argparse dest names
  search_space:  {name: {type: float|categorical, ...}} — see align/configs/sweep.yaml

Objective: best val reward_acc reached during the trial (fast, computed on every
run regardless of --heldout-eval — see align/configs/sweep.yaml for why this was
chosen over the de novo Spearman for the search loop itself).

The study is persisted to a local sqlite DB, so it's resumable and inspectable
with `study.trials_dataframe()` or `optuna-dashboard sqlite:///align/dpo_out/optuna_study.db`.

Usage (from rl_esm/):
    python align/sweep_dpo.py --config align/configs/sweep.yaml --n-trials 30
    python align/sweep_dpo.py --n-trials 2 --study-name smoke_test   # tiny sanity check
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import optuna
import torch
import yaml
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

import train_dpo

HERE = Path(__file__).resolve().parent


def suggest(trial: optuna.Trial, name: str, spec: dict):
    t = spec["type"]
    if t == "float":
        return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
    if t == "int":
        return trial.suggest_int(name, spec["low"], spec["high"], log=spec.get("log", False))
    if t == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    raise ValueError(f"unknown search_space type {t!r} for {name!r}")


def fmt(v) -> str:
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def trial_argv(fixed: dict, params: dict, run_name: str) -> list[str]:
    """fixed/params keys must match train_dpo.build_args' argparse dest names."""
    argv = []
    for k, v in {**fixed, **params}.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                argv.append(flag)  # store_true flags: omit when False
        else:
            argv += [flag, str(v)]
    argv += ["--run-name", run_name]
    return argv


def make_objective(cfg: dict, config_path: Path, exp_name: str):
    fixed, space = cfg["fixed"], cfg["search_space"]

    def objective(trial: optuna.Trial) -> float:
        params = {name: suggest(trial, name, spec) for name, spec in space.items()}
        run_name = f"trial{trial.number:03d}_" + "_".join(f"{k}{fmt(v)}" for k, v in params.items())
        args = train_dpo.build_args(trial_argv(fixed, params, run_name))
        args.config = config_path  # provenance only — argv above already carries every value
        # group all trials of this study under one experiment folder, unless the
        # config's `fixed` block already pins an exp_name
        if not args.exp_name:
            args.exp_name = exp_name

        print(f"\n{'=' * 80}\n[optuna trial {trial.number}] {params}\n{'=' * 80}")
        try:
            metrics = train_dpo.run(args, trial=trial)
        finally:
            torch.cuda.empty_cache()
            gc.collect()
        return metrics["best_reward_acc"]

    return objective


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=HERE / "configs" / "sweep.yaml")
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--timeout", type=int, default=None, help="wall-clock budget in seconds")
    p.add_argument("--study-name", default="dpo_sweep")
    p.add_argument("--storage", default=None,
                   help="defaults to sqlite:///align/dpo_out/optuna_study.db")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    assert args.config.exists(), f"missing {args.config}"
    cfg = yaml.safe_load(args.config.read_text())
    assert "fixed" in cfg and "search_space" in cfg, f"{args.config} needs top-level fixed/search_space keys"

    train_dpo.OUT_DIR.mkdir(parents=True, exist_ok=True)
    storage = args.storage or f"sqlite:///{train_dpo.OUT_DIR / 'optuna_study.db'}"

    study = optuna.create_study(
        study_name=args.study_name, storage=storage, load_if_exists=True,
        direction="maximize", sampler=TPESampler(seed=args.seed), pruner=MedianPruner())
    study.optimize(make_objective(cfg, args.config, args.study_name),
                   n_trials=args.n_trials, timeout=args.timeout)

    print(f"\n=== best trial: #{study.best_trial.number} ===")
    print(f"  reward_acc = {study.best_value:.4f}")
    print(f"  params     = {study.best_trial.params}")

    best_cfg = {**cfg["fixed"], **study.best_trial.params}
    out_path = HERE / "configs" / "best_sweep_config.yaml"
    header = (
        f"# Winning hyperparams from align/sweep_dpo.py "
        f"(study={args.study_name!r}, trial #{study.best_trial.number}, "
        f"reward_acc={study.best_value:.4f}).\n"
        f"# Full-data confirmatory run (overrides the sweep's shrunk max_pairs/heldout_eval):\n"
        f"#   pixi run python align/train_dpo.py "
        f"--config align/configs/best_sweep_config.yaml --max-pairs 0 --heldout-eval\n\n")
    out_path.write_text(header + yaml.safe_dump(best_cfg, sort_keys=False))
    print(f"\nWrote winning config → {out_path}")


if __name__ == "__main__":
    main()
