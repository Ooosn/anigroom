# R073 Budget-Normalized Trusted Ownership

Status date: 2026-08-30.

Status: implementation and local validation in progress. No H100 result is
accepted by this document.

## Question

R072 keeps the accepted V7 trusted support but uses its raw q95-normalized
confidence as the optimizer multiplier. The mean training gate is only
`0.06993`, so the run loses root motion, opacity fitting, lifecycle growth, and
view-09 quality while remaining granular. Is the trusted support useful when
selection is separated from optimization strength?

## Single Method Change

R073 inherits R072 exactly and changes only `VIEW_GATE_NORMALIZATION` from
`raw_q95` to `equal_owner_budget`.

For guide `g`, let `k_g` be the number of nonzero trusted owners among the
concrete `N=30` training views. Every owner receives multiplier

`N / k_g`

and every non-owner receives zero. Therefore uniform training-view sampling
has expected multiplier one for every guide with at least one owner. A guide
with no owner remains zero and relies on the same K8 surface interpolation and
view-independent regularizers as R072.

This is importance weighting, so multipliers may exceed one. The Panda target
has median/P90 owner counts `7/10`; positive multipliers have
P50/P90/P95/P99/max `3.75/6/7.5/15/30`. The rule has no tuned scale or cap.

## 0-3k Safety Gate

`configs/r073_budget_normalized_ownership_0_3k_gate.env`

The first H100 run stops at 3k because `169` guides have only one training
owner and therefore receive a 30x multiplier when that owner is sampled.
Acceptance requires finite gradients and Adam state, no lifecycle explosion,
valid memory/checkpoint reload, and a full-resolution view-09 crop materially
closer to the clean single-view control than both R068 and R072. A 6k/9k run is
authorized only after that visual gate.
