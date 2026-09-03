# Post-V8 Global Direction-Field Attempt

Status date: 2026-09-03.

Status: completed fixed-input diagnostic. The selected `smooth_weight=3`,
`orientation_barrier_weight=10` candidate is pending asset/user visual
acceptance and has not been integrated or trained. Formal V8 remains the
accepted training parent.

## Parent and motivation

The formal V8 Panda and white-tiger parents are immutable inputs:

- Panda: `5cb76945adb034e9666bfc98ae05647062d7ac4e3609e68162e561e4eebd54b1`.
- White tiger: `92a6d496aa39e85272f35668967f82d34df7f884681ade4e336c07256b47a3d7`.

The existing post-V8 angle/lift candidate did not pass the user-local arrow
visual. Before this global solve, local continuous stronger-neighbor
propagation improved aggregate metrics but left top/right screen conflicts;
its code was discarded. Its artifacts were
`D:/RTS/_tmp/panda_continuous_confidence_flow_20260903_attempt1` and
`D:/RTS/_tmp/white_continuous_confidence_flow_20260903_attempt1`. The sixth
cycle was accepted, but the seventh was rejected despite `1039-1500` locally
eligible roots. This establishes a local-greedy/global-gate deadlock rather
than a lack of locally eligible updates.

## Isolated algorithm and gates

The new diagnostic is a joint tangent-angle/log-lift global field solve. Its
dimensionless objective is

`L = E_data / E_data0 + smooth_weight * mean(E_surface / E_surface0, E_tangent / E_tangent0, E_lift / E_lift0)`.

Here `E_data` is confidence-weighted axial reprojection data, and the three
connection terms are final surface, transported tangent, and log-lift field
energies. The solve uses a parallel-transport graph, an orientation barrier on
edges that were previously nonnegative and nonsevere, deterministic Adam, and
powers-of-two backtracking.

An update is accepted only when all of the following are nonincreasing:

- data, surface, tangent, and lift energies;
- edge P95, P99, top-1%-CVaR, and maximum;
- negative/severe edge counts and negative/severe root counts.

It must also introduce zero newly bad roots. There is no species, region, view,
or image rule and no protected owner.

## Honest attempt ledger

| Attempt | Result | Decision |
| --- | --- | --- |
| Local continuous stronger-neighbor propagation | Aggregate metrics improved; top/right screen conflicts remained. Cycle 6 accepted and cycle 7 rejected with `1039-1500` locally eligible roots. | Rejected; code discarded. |
| Existing post-V8 angle/lift candidate | Did not pass user-local arrow visual. | Rejected as a usable target. |
| First no-barrier global-field sweep | No setting was accepted cross-sample. | Rejected. |
| Edge-identity/root-support gate sweeps | No setting was accepted cross-sample. | Rejected. |
| Orientation-barrier sweep | `10/10` runs passed the generic gate. | Retained for the shared follow-up. |
| Shared smooth sweep with barrier `10` | `smooth=3` gives the strongest Panda correction while white passes every generic gate with the same settings and automatic backtracking. | Selected diagnostic candidate; this is not claimed to be the automatic min-ranking winner. |

The selected setting is therefore a diagnostic choice pending physical-asset
and user acceptance, not an automatic ranking claim.

## Selected metrics

Values are initial → final. Edge-tail and direction-change entries are in
degrees; `negative/severe` pairs are counts.

| Sample | Data | Surface | Tangent | Lift | Edge P95/P99 | Negative/severe edges | Negative/severe roots | Direction change P95/max | New bad roots |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Panda | `0.151728958→0.134471998` | `0.081285164→0.022716768` | `0.092970185→0.026532158` | `0.047688868→0.009225478` | `55.3008/94.7821→25.8670/44.9140` | `365/10→8/0` | `264/16→11/0` | `44.788/68.655` | `0` |
| White tiger | `0.210384637→0.208973378` | `0.102271296→0.093914248` | `0.111925289→0.102144413` | `0.035596021→0.031667717` | `60.1170/87.1834→57.3577/83.7524` | `248/45→230/35` | `204/47→194/41` | `3.467/7.857` | `0` |

The selected edge top-1%-CVaR/max also decrease: Panda
`113.503/157.405→59.386/134.422` and white tiger
`109.570/176.162→105.917/171.530`. No newly bad edge or root was introduced.

## User-local arrow QA

In the 71-root user QA, the V8 incident-3D-max median/P95/max changes from
`50.477/80.064/88.169` to `15.955/44.632/53.527`; `67` roots improve by more
than `1°`, while `4` worsen. The `33` visible fixed-length screen arrows
remain mixed and the top group still appears reversed. This is explicitly not
hidden by the aggregate field metrics.

Re-running the existing V7 signs flips `10` Panda roots and cleans some center
arrows, but the top cluster remains. On white tiger, post-V8 lift worsens from
`0.031668` to `0.038818`, so reorientation is rejected. The selected candidate
is the pre-reorientation global field.

## Selected artifacts and provenance

Selected target candidates:

- Panda:
  `D:/RTS/_tmp/global_direction_field_smooth_barrier10_sweep_20260903/panda_s3/candidate_target.npz`, SHA-256
  `84a2637a74762a67f222c53f464872145d4a2a5ed1fe567a44329d23792ac493`.
- White tiger:
  `D:/RTS/_tmp/global_direction_field_smooth_barrier10_sweep_20260903/white_s3/candidate_target.npz`, SHA-256
  `c27647a2d171d0a5d8c61bb3bd91742094522ab761524d2c4107b0623b997849`.

The accompanying final-direction hashes are Panda
`ff30158f161ca633db2133f9ad18dc44e29eefb7dea893586bb6e5176ba979b9` and white
tiger `6f104dc98b532986eec28e7e37934f025390cbea103904e00389f08bd39ea43c`.
Deterministic reruns are exact in
`D:/RTS/_tmp/global_direction_field_selected_determinism_20260903`.

Panda visual evidence:

- Full overlay:
  `D:/RTS/_tmp/panda_global_field_s3_b10_arrows_view09_20260903/direction/view09_shell_cleaned_3d_arrows_overlay.png`, SHA-256
  `7fa7f95618f73fce1dd2b471ca55aa1dccceaf0bf83c3cbce8916f4d35425e60`.
- Exact crop:
  `D:/RTS/_tmp/panda_global_field_s3_b10_arrows_view09_20260903/direction/view09_user_region_crop_4x.png`, SHA-256
  `5f66705142cf5bc4380c59ecc4d3a352c4311a1ef3983a3238cae6f3a4fd85bf`.

The matched selected white-tiger view09 overlay is
`D:/RTS/_tmp/white_global_field_s3_b10_arrows_view09_20260903/direction/view09_shell_cleaned_3d_arrows_overlay.png`,
SHA-256
`52d97bbfb3d91552e2576b659ee8511aaea46193ef5c9bf0b6314934e1f8bbc4`.
The earlier white visualization under
`D:/RTS/_tmp/white_global_field_b10_arrows_view09_20260903` remains a distinct
`smooth=1` diagnostic and must not be used as selected-candidate evidence.

The isolated implementation files are
`anigroom/flow/global_direction_field_refinement.py`,
`tools/diagnose_global_direction_field_refinement.py`, and
`tests/test_global_direction_field_refinement.py`. Final validation passes
`17` focused tests and the complete repository suite at `729 passed` with the
existing `14` Matplotlib/pyparsing deprecation warnings. No trainer or
configuration integration was performed.

## Established basis and decision

The approach follows the established global-field direction literature:
[Knöppel et al., *Globally Optimal Direction Fields*](https://doi.org/10.1145/2461912.2462005)
and [Vaxman et al., *Directional Field Synthesis, Design, and Processing*](https://doi.org/10.1111/cgf.12864).
The relevant basis is dense soft constraints plus Dirichlet-style field energy;
under curvature, local path propagation can be inconsistent because of
holonomy, motivating a global solve.

Formal V8 remains the accepted training parent. The selected global-field
candidate is not integrated or trained until it passes the user/physical-asset
visual gate.
