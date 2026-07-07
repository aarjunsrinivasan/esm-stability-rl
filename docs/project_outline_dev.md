# RL-Aligning a Protein Language Model — Project Spec

**One-liner:** Align a protein language model (ESM-C) to a *measurable* fitness objective
two ways — **offline DPO vs. online GRPO** — and run an honest head-to-head: which
optimizes better, which stays in-distribution, and which reward-hacks. Validated against
real experimental data.

**Why this project:** LoRA fine-tuning and TRL are commodities — everyone's portfolio
has them. The differentiating signal for an RL/post-training research role (e.g. CZI
Biohub) is: *can you design the reward, build the eval, and distrust your own results?*
This project is built entirely around that.

**The novel angle (grounded in the literature):** ProteinDPO (Widatalla, Rafailov & Hie,
2024) already aligned a protein generative model to stability with **DPO**. A controlled
**DPO-vs-GRPO** comparison on the same objective — with a shared-latent-space
reward-hacking analysis — is *not* standard and directly extends that line of work. That
comparison is the paper.

---

## The core idea

Frame it as **in-silico directed evolution**: steer ESM-C toward higher-fitness sequences
while staying biologically plausible. Do it **two ways** and compare:

```
  OFFLINE:  Megascale ΔG ──▶ preference pairs (A≻B) ──▶ DPO ──▶ aligned policy
  ONLINE:   policy ──samples──▶ reward oracle ──reward──▶ GRPO (reward − β·KL) ──▶ policy
                ▲_______________________________________________________|

  held-out checks (NOT in either reward):  ESMFold pLDDT · base-model perplexity
                                            · ProteinGym experimental ground truth
```

The headline deliverable is not "reward went up" — it's the **DPO-vs-GRPO comparison plus
a reward-hacking analysis**, with held-out oracles and ground-truth validation. If a
policy games its reward, the held-out signals won't follow.

### Correction that sharpens the reward (from the ESM3 paper, App. A.1.4.4)
Megascale is **not** ESM pretraining data — the paper uses it to *probe frozen ESM-C
representations*, and shows a **linear ridge probe already predicts ΔG at Spearman
0.68–0.8, on par with FoldX/Rosetta, from sequence alone.** Two consequences:

1. **The reward oracle is a solved, cheap component, not the contribution.** Use a ridge
   probe on the frozen ESM-C penultimate layer as your v1 reward — no LoRA head needed.
   The contribution is the *alignment comparison*, not the predictor.
2. **Shared-latent-space hacking is the real risk to headline.** An ESM-C policy
   optimizing an ESM-C-*derived* reward is the model grading itself — a textbook setup
   for adversarial directions that spike the reward without real stability. This makes the
   **held-out ESMFold + experimental checks non-negotiable** and makes your hacking story
   *more* compelling.

---

## Concrete instantiation

| Component | Choice |
|---|---|
| **Policy / base model** | ESM-C 300M (small, fast to sample, fits one GPU). Don't reach for ESM3/bigger until it works. |
| **Task** | Optimize a starting protein/family toward higher predicted stability (ΔG/ΔΔG). |
| **Reward (v1)** | Ridge probe on **frozen** ESM-C penultimate layer → ΔG (paper shows Spearman ~0.68–0.8). Cheap, scored every step. |
| **Held-out oracle #1** | ESMFold pLDDT (foldability) — zero training, scored periodically on held-out samples. |
| **Held-out oracle #2** | Base-model perplexity / naturalness — detects drift off the data manifold. |
| **Ground truth** | ProteinGym experimental DMS measurements on proteins the reward never saw. |
| **Algorithms** | **DPO** (offline, Megascale pairs) **vs GRPO** (online, oracle reward) — both via HuggingFace TRL, KL to frozen base. |

---

## Core experiment: DPO vs GRPO

Same base model, same objective, same held-out evals — only the alignment method changes.

| | DPO (offline) | GRPO (online) |
|---|---|---|
| Data | preference **pairs** (mutant A ≻ B by ΔG) | a **reward function** (the ridge probe) |
| Loop | contrastive loss vs frozen ref; no sampling | policy samples → oracle scores → group-relative advantage |
| Strength | stable, cheap, aligns the distribution | **explores/designs novel** seqs; can use **non-differentiable** rewards (ESMFold pLDDT!) |
| Weakness | bounded by pair distribution; no exploration | compute-heavier; reward-hacking prone |
| Best for | "prefer more stable variants in known families" | "design novel optimized sequences de novo" |

**What you measure on the held-out set for each method:** final ΔG gain · sequence
**novelty/diversity** (distance from training seqs) · naturalness (Δ perplexity) ·
foldability (ESMFold pLDDT) · agreement with ProteinGym ground truth · **reward-hacking
gap** (train-reward up while held-out signal flat/down).

**Hypothesis worth stating up front:** GRPO reaches higher reward but hacks harder (esp.
with a shared-latent reward) and drifts off-manifold; DPO is safer but plateaus at the
pair distribution. The plot that shows *where each one breaks* is the deliverable.

---

## Data

| Purpose | Dataset | Size | Notes |
|---|---|---|---|
| Reward + DPO pairs | **Tsuboyama 2023 "mega-scale" ΔG** | ~776k measurements, 331 natural + 148 de novo domains, 40–72 aa | **Used by ESM only as a probe/eval, not pretraining** — so a legit supervised reward. Short domains keep ESMFold cheap. |
| Ground-truth eval | **ProteinGym** substitution DMS | ~2.5M mutants, 217 assays | Use assays on proteins *excluded* from the reward. The anti-hacking proof. |
| Curated sanity check | **FireProtDB** (homolog-free, ThermoMPNN split) | ~2.6k global ΔΔG | ESM's own out-of-distribution generalization set; free extra held-out. |

**Non-negotiable design choice:** split by *structural* similarity (Foldseek clusters, as
ESM/ProteinDPO do), not randomly. Fit reward/DPO on some families, evaluate discoveries on
structurally disjoint ones. This is what makes the eval credible.

### Datasets ESM did *not* finetune on — each a project of its own
ESM evaluates stability (Megascale/FireProt), DMS fitness (ProteinGym), function
(EC-CATH). The gaps below are where general PLM pretraining is *sparse* or measures a
property the MLM loss doesn't capture — i.e. where finetuning actually pays:

| Dataset | Why finetune | What it buys |
|---|---|---|
| **OAS** (antibody repertoires) | PLMs underperform on hypervariable CDRs, sparse in pretraining | specialist antibody LM (design, humanization) |
| **SKEMPI** (binding ΔΔG) | folding stability ≠ binding affinity | binder design, interface-mutation effects |
| **eSOL / aggregation** (expression, solubility) | a stable protein may not express | developability / manufacturability |
| **Enzyme kcat/Km** (BRENDA/SABIO) | EC *class* ≠ catalytic *rate* | functional optimization, directed evolution |
| **MHC / immunogenicity** (IEDB) | not in the evolutionary signal | de-immunizing therapeutics |
| **Multi-mutant landscapes** (GB1) | Megascale is mostly single mutants | epistasis / higher-order interactions |

---

## Structure-based variant — which dataset?

If you want the objective to be **3D structure** rather than a scalar sequence property,
the clean framing is **designability via self-consistency**: policy proposes a sequence,
a folding model (ESMFold) predicts its structure, reward = **scTM** (TM-score of the
predicted fold vs. the intended target backbone). It's the standard design metric
(RFdiffusion / ProteinMPNN use scTM > 0.5) and needs **no labels beyond backbones**. The
DPO-vs-GRPO comparison carries straight over — and because the folding reward is
**non-differentiable**, GRPO is the natural fit (DPO would need pairs like
high-scTM ≻ low-scTM).

**The one hazard that dominates a structure project: leakage.** ESM3/ESMFold were trained
on the PDB, so any natural structure you condition on is probably *in* the training set.
Your dataset choice is really a choice about *how you prove generalization.*

| Dataset | Size | Best for | Leakage risk | Verdict |
|---|---|---|---|---|
| **CATH-S40 / ProteinMPNN CATH4.2 splits** | ~20k domains, public splits | inverse-folding targets + native-seq-recovery baseline | high (in PDB) — but scTM self-consistency tolerates it | **Primary conditioning set.** Standard, public, reproducible. |
| **CAMEO (rolling) / CASP15–16** | 100s of targets | leakage-free eval (post-cutoff release) | **none** (temporal holdout) | **Primary held-out eval.** Small but clean — the credibility anchor. |
| **Tsuboyama 148 de novo domains** | 148, with ΔG *and* structure | leakage-free validation; ties to the sequence project | **none** (not in natural evolution) | **Hidden gem** — same dataset as the sequence arm, so both projects share infra. |
| **Rocklin 2017 mini-proteins** | ~15k designed, stability-labeled | small designable scaffolds | low (designed) | Good extra de-novo validation. |
| **AlphaFold DB / ESM Atlas** | 200M+ predicted | scale, unlabeled backbones to condition on | n/a (predictions) | Only if you need volume; predictions ≠ ground truth. |
| **PDBbind / SKEMPI / DIPS** | 10k–100k complexes | **binder/interface design** with an ipTM reward | high | Pick this only if the project is *binding*, not folding. |

**Recommendation:** condition on **CATH-S40 backbones**, reward = **ESMFold scTM**,
prove generalization on **CAMEO-recent + the 148 Tsuboyama de novo domains** (zero
leakage). Native sequence recovery on CATH is your supervised sanity anchor. This reuses
the *entire* sequence-project pipeline — only the reward function changes from ΔG-probe to
scTM — so the two variants are one codebase, and "designability reward hacks by generating
low-complexity sequences ESMFold over-confidently folds" is a real, documented failure
mode you can showcase.

**Compute note:** a structure reward puts ESMFold *in* the loop, which is the expensive
part (~1–5 s/seq). Cap target length at ≤128 aa (CATH domains / de novo are short), batch
folding, and score scTM on a rollout subsample — budget ~2–3× the sequence project's
GPU-hours.

---

## Compute & timeline

**Budget: ~3 focused weeks, ~60–150 GPU-hours (~$100–300 cloud, or ~1 week of one GPU).**
Much cheaper than gym-RL instincts suggest, because ESM-C 300M is tiny and the reward
head is fast.

| Phase | Days | GPU-hrs | Deliverable |
|---|---|---|---|
| 0. Scope + baselines | 2 | ~2 | Task frozen; ridge-probe reward (held-out Spearman reported); base-model samples |
| 1. DPO arm | 2–3 | 8–20 | Megascale preference pairs; DPO-aligned policy; held-out fitness gain |
| 2. GRPO arm | 4–5 | 30–80 | Online GRPO on the probe reward; KL-to-base logged |
| 3. Comparison + hacking analysis | 3–4 | 25–55 | DPO vs GRPO on every held-out metric; KL sweep; ProteinGym validation; hacking gap |
| 4. Writeup + polish | 2–3 | ~0 | Blog post **with negative results**; clean reproducible repo |

**Keep ESMFold out of the inner RL loop** — it's the throughput killer (~1–5 s/sequence).
Reward on the fast head every step; score with ESMFold only periodically on held-out
samples. That one choice keeps the whole thing on a single GPU.

---

## Suggested repo structure

```
rl-esm-design/
├── README.md
├── data/
│   ├── build_pairs.py              # Megascale -> DPO preference pairs, Foldseek-split
│   └── proteingym_eval_set.py      # held-out ground-truth mutants
├── reward/
│   ├── fit_probe.py                # ridge probe on frozen ESM-C penultimate -> ΔG
│   └── eval_probe.py               # held-out Spearman/RMSE (GATE before alignment)
├── align/
│   ├── policy.py                   # ESM-C policy + frozen ref
│   ├── train_dpo.py                # TRL DPO on Megascale pairs (offline arm)
│   ├── train_grpo.py               # TRL GRPO on probe reward (online arm)
│   └── config.yaml
├── analysis/
│   ├── compare.py                  # DPO vs GRPO across every held-out metric
│   ├── hacking_report.py           # train-reward vs held-out (pLDDT, ppl) divergence
│   ├── kl_sweep.py                 # β sweep: reward vs naturalness frontier
│   └── validate_proteingym.py      # do discovered mutations agree with real DMS?
└── writeup/
    └── report.md                   # the story, with plots and negative results
```

---

## What makes it "click" with hiring managers

1. **A real comparison, not a demo.** DPO vs GRPO on one objective, controlled — that's a
   result, not a tutorial rerun. It extends published work (ProteinDPO) instead of copying it.
2. **You did real RL, not SFT** — KL control, credit assignment, reward hacking discussed
   and demonstrated, not just invoked.
3. **Structure-split eval + experimental ground truth** — proves generalization, not
   memorization; and you *knew* Megascale was a probe set, not pretraining.
4. **The money shot:** "GRPO reached higher reward but hacked the shared latent space;
   here's the plot of train-reward vs. held-out ESMFold diverging, the KL sweep that fixed
   it, and where DPO plateaued instead." That reads like a research log — which is the job.

A repo with only green checkmarks reads like homework. A writeup with an honest negative
result reads like a researcher.

---

## Failure modes to instrument from day one (don't discover them in week 3)

- **Reward too easy to hack** → policy collapses to a few garbage-but-high-scoring
  sequences. This is *good* — it's your story. Log naturalness + pLDDT from step 1.
- **Oracle overfits** → your whole reward is noise. Report held-out Spearman *before*
  touching RL; if <0.4, fix the oracle first.
- **No ground truth** → you have "reward went up" and nothing else. The ProteinGym
  validation is the credibility anchor; wire it early.

---

## Mapping to the Biohub JD

| JD requirement | How this project answers it |
|---|---|
| "post-training systems: RL, reward modeling" | The entire project |
| "evaluation frameworks grounded in real biological outcomes" | Identity-split + ProteinGym validation |
| "comfort with ambiguity, rapid iteration, loosely-defined problems" | Reward-hacking diagnosis and KL tuning |
| "strong communication across technical and scientific audiences" | The writeup with negative results |

---

## One-sentence version to internalize

*The library gives you the optimizer; the job is filtering for people who can design the
reward and distrust their own results. This project proves you do both.*

---

### Portfolio note
This composes with the **SARS-CoV-2 interactome × ESM-C** hackathon project (see
`krogan_hackathon/`): both are protein/ESM-centric and both center on *rigorous evaluation
of biological foundation models*. Present them under one banner rather than as unrelated
one-offs.
