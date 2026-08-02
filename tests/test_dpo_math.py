"""The DPO loss core. `dpo_terms` decides both what the model learns and what
`kl_drift` reports, so its signs are load-bearing well beyond training."""
from __future__ import annotations

import math

import pytest
import torch

from align.train_dpo import dpo_terms, split_pair

LOG2 = math.log(2.0)


def test_identical_policy_and_reference_is_the_no_preference_fixed_point():
    x = torch.tensor([0.5, -2.0, 7.25])
    loss, margin, d_chosen = dpo_terms(x, x, x, x, beta=0.1)

    # margin == 0 is the decision boundary, i.e. exactly 50% reward accuracy.
    assert torch.allclose(margin, torch.zeros(3))
    assert torch.allclose(d_chosen, torch.zeros(3))
    assert loss.item() == pytest.approx(LOG2)


def test_preferring_chosen_lowers_the_loss_and_preferring_rejected_raises_it():
    ref_c = ref_r = torch.tensor([0.0])

    good_loss, good_margin, _ = dpo_terms(torch.tensor([1.0]), torch.tensor([-1.0]),
                                          ref_c, ref_r, beta=0.1)
    bad_loss, bad_margin, _ = dpo_terms(torch.tensor([-1.0]), torch.tensor([1.0]),
                                        ref_c, ref_r, beta=0.1)

    assert good_margin.item() > 0 and good_loss.item() < LOG2
    assert bad_margin.item() < 0 and bad_loss.item() > LOG2
    # The loss is symmetric about the boundary, so a mirrored pair mirrors exactly.
    assert good_margin.item() == pytest.approx(-bad_margin.item())


def test_margin_scales_linearly_in_beta():
    pol_c, pol_r = torch.tensor([2.0]), torch.tensor([0.5])
    ref_c, ref_r = torch.tensor([1.0]), torch.tensor([1.0])

    _, m1, _ = dpo_terms(pol_c, pol_r, ref_c, ref_r, beta=0.1)
    _, m2, _ = dpo_terms(pol_c, pol_r, ref_c, ref_r, beta=0.2)

    assert m2.item() == pytest.approx(2 * m1.item())


def test_d_chosen_is_the_signed_drift_logged_as_kl_drift():
    """§6 of the README reads a *negative* kl_drift as the policy making chosen
    sequences less likely than the reference does. That reading depends on
    d_chosen being pol_c - ref_c and not the reverse."""
    pol_c, ref_c = torch.tensor([1.0, -3.0]), torch.tensor([4.0, -1.0])
    _, _, d_chosen = dpo_terms(pol_c, torch.zeros(2), ref_c, torch.zeros(2), beta=0.1)

    assert torch.allclose(d_chosen, pol_c - ref_c)
    assert (d_chosen < 0).all()      # policy below reference -> negative drift


def test_only_the_relative_margin_matters_not_absolute_likelihood():
    """The reason alignment costs perplexity (README section 6): shifting both
    sides down by the same amount leaves the loss untouched, so nothing in the
    objective defends absolute likelihood."""
    base_loss, _, _ = dpo_terms(torch.tensor([2.0]), torch.tensor([1.0]),
                                torch.tensor([0.0]), torch.tensor([0.0]), beta=0.1)
    shifted_loss, _, _ = dpo_terms(torch.tensor([-98.0]), torch.tensor([-99.0]),
                                   torch.tensor([0.0]), torch.tensor([0.0]), beta=0.1)

    assert base_loss.item() == pytest.approx(shifted_loss.item())


def test_split_pair_round_trips_a_stacked_chosen_rejected_batch():
    chosen, rejected = torch.tensor([1.0, 2.0, 3.0]), torch.tensor([-1.0, -2.0, -3.0])
    back_c, back_r = split_pair(torch.cat([chosen, rejected]), len(chosen))

    assert torch.equal(back_c, chosen)
    assert torch.equal(back_r, rejected)
