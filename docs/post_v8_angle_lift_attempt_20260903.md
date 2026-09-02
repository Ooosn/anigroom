# Post-V8 Angle/Lift Attribution Attempt

Status date: 2026-09-03.

Status: completed fixed-input attempt series. One cross-sample candidate passes
the automatic four-energy gate; it remains a target candidate and has not been
used for training.

## Reopened Question

The canonical view09 complete-direction arrows can look locally disordered even
when the transported 3D field is smoother. A screen arrow depends on both the
surface angle and normal/tangent ratio:

`screen = project(normalize(a(theta) + rho * normal))`.

On the final Panda view09 probe, `1360` roots are visible. Complete-direction
versus tangent-only screen angle has median/P95/P99
`12.09/44.25/98.39` degrees and correlates with `rho` at `0.585`. Among `612`
screen-neighbor edges above 45 degrees, `148` have transported 3D angle below
20 degrees, while `237` are genuinely above 45 degrees in 3D. Of `256` roots
incident to those screen conflicts, `51` have greater screen-angle sensitivity
to `log(rho)` than to tangent angle and `13` exceed a `5x` sensitivity ratio.

Therefore screen disorder cannot be assigned wholesale to the tangent angle.

## Shared Automatic Gate

Every arm starts from the same formal V8 field, `[33,4500]` per-view confidence
matrix, V8 joint root reliability, graph, optimizer settings, backtracking, and
outer stop. One complete BA-plus-V8-propagation cycle is accepted only if all
four quantities are nonincreasing and at least one improves strictly:

- formal multiview residual;
- complete-direction connection energy;
- transported tangent-axis connection energy;
- scalar `log1p(rho)` edge energy.

No species, region, view, or image-coordinate rule enters the solve.

## Attempt Ledger

| arm | Panda result | reason / next action |
| --- | --- | --- |
| `theta_only` | one cycle accepted | rejected: view09 screen >45 edges `584 -> 612`, true 3D >45 edges `324 -> 350` |
| `rho_only` | zero cycles accepted | every scale roughens the lift field |
| `joint` | one small Panda cycle | rejected cross-sample: white accepts zero because lift energy rises |
| `theta_observable` | one Panda cycle | rejected: Jacobian weighting still yields view09 `584 -> 655` screen >45 edges |
| `joint_observable` | zero Panda cycles | split gradients cannot satisfy data and lift gates together |
| `joint`, lift-delta smooth `0.01` | one Panda cycle | rejected: P95 change `44.01 deg`, view09 roughness increases |
| `joint`, lift-delta smooth `0.1` | zero Panda cycles | no Pareto-acceptable scale |
| `joint`, lift-delta smooth `1.0` | one Panda and one white cycle | selected cross-sample candidate |

The selected arm regularizes only the spatial variation of the learned
`delta log(rho)`, not the inherited physical lift field. Panda and white use
the identical dimensionless weight `1.0`.

## Selected-Arm Results

| sample | accepted scale | data gain | complete-field gain | tangent-field gain | lift-field gain | P95 direction change | cycle 2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Panda | `0.0625` | `1.422%` | `4.774%` | `4.193%` | `2.656%` | `6.435 deg` | rejected |
| white tiger | `0.0625` | `2.963%` | `0.303%` | `0.440%` | `2.492%` | `4.855 deg` | rejected |

The propagation stage changes `240/34` Panda/white roots. Both report zero new
severe edges. Fixed `00/09/18/27` screen projections have small mixed changes,
while transported 3D and screen tails improve in most views. In the user-relevant
Panda view09, screen edges above 90 degrees move `204 -> 195`; above 45 degrees
move `584 -> 581`. The transported 3D >45 count remains `324`, with P95
`38.94 -> 39.31` degrees. The candidate is therefore conservative rather than a
claim that every projection statistic is monotone.

Independent reruns reproduce direction SHA-256 exactly:

- Panda: `8fdc8fa5fa884a46541f2245b08c9a925601fa35f9ac600c98d05356b0633114`;
- white tiger: `7d78a5860b40bbbe35881cebfee1c02ee8f97ab8d8ceaad6a5a118bd76c8bdad`.

## Artifacts

Panda selected arm:

- report:
  `D:/RTS/_tmp/panda_post_v8_angle_lift_20260903_joint_smooth_1_finalsource/post_v8_refinement.json`,
  SHA-256 `20e5862bb8e2efb1c2e1b80697ed2c98e176b2a1c9d8977d74ce8da098899e48`;
- target:
  `D:/RTS/_tmp/panda_post_v8_angle_lift_20260903_joint_smooth_1_finalsource/candidate_target.npz`,
  SHA-256 `211436456453f06393e5aadc858a59593853134e2f5645fa4e36b1a603d57393`;
- manifest / runner-result SHA-256:
  `5c6a9d5f43b7994ebe1996300936b4b01eeaaa7708bbaa01f4fd8f672525a3ef`
  / `7bcd0fb24d0fa8ff509d79094298e349500f9e9bce34a7c1b8427f64ed63e695`.

White selected arm:

- report:
  `D:/RTS/_tmp/white_post_v8_angle_lift_20260903_joint_smooth_1_finalsource/post_v8_refinement.json`,
  SHA-256 `5cdd83d2e46bc0b3a6600862e4965fc86f75c5f1dc63e3b4613db39270adae16`;
- target:
  `D:/RTS/_tmp/white_post_v8_angle_lift_20260903_joint_smooth_1_finalsource/candidate_target.npz`,
  SHA-256 `90816784d37c690718cc011ca86fb4ed13d9259ae7f4ce4bf178ef85a14b0501`;
- manifest / runner-result SHA-256:
  `e39069ae88408fb1a74e95275d07b68624a750f7ddc5ba6e264c93c6415a2516`
  / `f38c6e3b8dd6fe59f27e9c99937a6aefbdf111ee9549b9e67a2874daf6ac7bf1`.

Canonical Panda view09:

- direction surface:
  `D:/RTS/_tmp/panda_post_v8_angle_lift_joint_smooth1_canonical_view09_20260903/view09_direction_surface_external.png`,
  SHA-256 `0009d24255fec3154f6edf8b1a83ff03cea8b4663e0a0094a5829206892ef43b`;
- complete-direction arrows:
  `D:/RTS/_tmp/panda_post_v8_angle_lift_joint_smooth1_arrows_view09_20260903/direction/view09_shell_cleaned_3d_arrows_overlay.png`,
  SHA-256 `912240d0502aa419576b129d6a6ecbe170ea837a4ae4076bcec9a0bfc332b0da`.

Implementation remains isolated in `anigroom/flow/post_v8_refinement.py` with
the actual-input runner `tools/diagnose_post_v8_refinement.py`. Focused tests
cover tangent-only, lift-only, joint, parameter observability, confidence-zero,
and automatic stopping. The complete repository passes `712` tests with the
existing `14` Matplotlib dependency warnings.

## Decision

The earlier tangent-only candidate is superseded. Joint angle/lift refitting is
not robust unless lift updates are spatially regularized. The weight-1.0 arm is
the only cross-sample candidate from this series, but the mixed fixed-view
projection changes require user visual review before any training gate.
