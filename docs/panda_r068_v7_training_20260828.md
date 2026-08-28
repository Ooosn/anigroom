# Panda R068 + V7 Training

Status date: 2026-08-28.

Status: formal from-zero H100 training and checkpoint-native coverage audit
completed successfully.

## Contract

This run preserves the complete accepted R068 behavior and changes only the
clean-flow target from Panda V5 to the accepted Panda V7 global-directed-flow
target.

- training source commit:
  `58bba7b7ea66745cf79346aa8e7046b08b9ea3a5`;
- behavior config: `configs/r068_no_crossing_zero_curl_0_30k.env`;
- data, mesh, alignment, SDF, resolution, schedules, losses, learning rates,
  lifecycle, root counts, Gaussian conversion, appearance residual, curl,
  no-penetration, and crossing-disabled behavior are unchanged from the
  completed R068+V5 run;
- V7 flow SHA-256:
  `6a220f52b15ca996c88e71802d3309f9499ade442f79dc72300f1af12b5fa56f`;
- mesh SHA-256:
  `20bd4d3cd2c48c886e2df96d8f183e75e5bacb1f6ebe0bba3c392677550d6c20`;
- SDF SHA-256:
  `a8ddedc9cd4bea81d9cda83610f57dcf0c30b3e6cabf8554360f8936fad9b7ab`;
- start mode: strict from-zero `0..30000`, no checkpoint or optimizer resume;
- expected initial render/guide/secondary-guide roots:
  `400000 / 4500 / 20000`;
- formal runtime:
  `/home/wangyy/panda-r068-v7-runtime-20260828`;
- run id: `panda_r068_v7_0_30k_h100_20260828`.

Training runs on held qlogin job `127181739`, `pcg02i`, physical H100 GPU2.
Held job `127114669`, `pcg01i`, physical H100 GPU7 currently contains an
unrelated user process and is not touched. Neither allocation may be released
without an explicit user request.

## Reopened baldness concern

The user reported that the completed R068+V5 Panda appears bald in multiple
areas. This is not hidden by the new flow result.

The bounded old-checkpoint audit separates it from a simple global-opacity
failure, but does not yet identify one unique cause:

- at alpha threshold `0.5`, fixed-view eroded-interior low-alpha fraction is at
  most about `0.015%` rather than a large mesh-hidden hole population;
- visible view-09 root opacity has mean `0.9544` and median `0.9991`;
- the view-09 length map contains large connected low-length basins on the
  upper back/rump and smaller shoulder/head patches;
- about `10%` of PLY Gaussians have a conservative projected maximum radius
  below `0.5 px` in views 09/27;
- the 100k asset is sample-conditioned and cannot prove full-population root
  holes.

Local guide-length collapse, directional footprint divergence, subpixel
Gaussian support, arbitrary-view clipping, and 100k asset sampling remain
plausible contributors. Exact attribution requires checkpoint-native pure-fur
rerenders over multiple views. No Panda-specific floor or hidden density
parameter is introduced into this V7 single-variable run.

The read-only audit is under
`D:/RTS/_tmp/panda_r068_v5_baldness_audit_20260828`; its summary SHA-256 is
`a6358e2a552b25c7bc9447202187164f97849efb8e37c8a7d357da2994bf0793`.

## Coverage monitoring

To avoid accepting another visually bald result, checkpoint audits are fixed at
iterations `9000`, `20000`, and `30000`:

- render views `09` and `27` from the exact checkpoint;
- generate groom-attribute maps for views `09` and `32`;
- record effective length, root opacity, tip opacity, root/Gaussian counts, and
  hashes;
- inspect connected low-length regions rather than relying only on global
  means or composite PSNR;
- run the three audits sequentially after training exits, on the same H100, so
  no monitoring process competes with the formal child.

The audit is observational: it does not alter training or automatically stop
the child. Any method change requires a separate, species-independent
experiment.

## Execution

- qlogin: `127181739` remains `r`;
- compute host/device: `pcg02i`, physical GPU2;
- child PID: `1498310`;
- bootstrap log:
  `/home/wangyy/logs/panda_r068_v7_0_30k_h100_20260828.bootstrap.log`;
- formal output:
  `/home/wangyy/panda-r068-v7-runtime-20260828/outputs/panda_r068_v7_0_30k_h100_20260828`.

The first launch attempt set `INIT_MESH_TRANSLATION='0 0 0'`; the reviewed
generic launcher requires comma-separated `0,0,0`. Its mandatory test gate
therefore stopped at `294 passed / 1 failed` before iteration one. No checkpoint
or training output was produced. The exact failed runtime/log/PID are preserved
under
`/home/wangyy/panda-r068-v7-failed-preflight-translation-20260828`.
The corrected launch uses `0,0,0` and otherwise changes no contract field.

## Terminal result

The corrected from-zero run completed successfully on 2026-08-28.

- source commit: `58bba7b7ea66745cf79346aa8e7046b08b9ea3a5`;
- final log marker: `[stage1] exit_code=0`;
- final checkpoint:
  `/home/wangyy/panda-r068-v7-runtime-20260828/outputs/panda_r068_v7_0_30k_h100_20260828/checkpoint_030000.pt`;
- checkpoint size: `1071879266` bytes;
- checkpoint SHA-256:
  `fb8c52ab50c7a879e6f18d2d1b2fd12475b276be89b194913c0507a843dc0ec2`;
- independent strict schema check: pass (`checkpoint_version=9`,
  `iteration=30000`, `checkpoint_kind=stage1_full`, no frizz keys,
  `strand_crossing_support=false`, zero crossing weight/refresh, null active
  set, zero refresh iteration, empty history);
- final train/test composite PSNR/SSIM:
  `29.815145/0.949172` and `28.763426/0.933673`;
- final roots/preclip Gaussians: `669143 / 7891276`; the sampled training view
  at iteration 30000 keeps `3986110` after mesh-depth clipping;
- final logged peak allocated CUDA memory: `22407.746 MB`;
- final metric elapsed wall time: `15791.344 s`;
- final post-evaluation allocated/reserved memory:
  `5969.602 / 7640.0 MB`;
- formal log contains no traceback, OOM, runtime-error, killed, CUDA-error, or
  failure markers.

Both held qlogins remained `r` throughout: `127181739` on `pcg02i:/dev/nvidia2`
and protected `127114669` on `pcg01i:/dev/nvidia7`.

## Checkpoint-native coverage results

The three audits ran sequentially on the same idle physical GPU2 from the clean
source checkout at the training commit. Each iteration has its own attribute
and render outputs, `coverage_attribute_summary.json`, and
`coverage_hashes.sha256` under
`/home/wangyy/panda-r068-v7-runtime-20260828/coverage_monitor/iter_XXXXXX`;
execution logs are under
`/home/wangyy/panda-r068-v7-runtime-20260828/coverage_monitor/logs`.

The connected-region diagnostic defines low length as visible-root length at or
below the per-view p10 and uses 8-connected projected pixels. It is an
observational baldness diagnostic, not a training threshold.

| checkpoint | view | visible roots | length mean / p10 | root opacity mean / fraction <=0.5 | tip opacity mean / fraction <=0.5 | largest low-length component |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 009000 | 09 | 331958 | 0.010791 / 0.009200 | 0.974779 / 2.479% | 0.957335 / 3.895% | 5669 roots, 1.708%, bbox `[281,407]-[405,517]` |
| 009000 | 32 | 272151 | 0.010890 / 0.009071 | 0.968188 / 3.110% | 0.946830 / 4.807% | 2996 roots, 1.101%, bbox `[891,153]-[985,192]` |
| 020000 | 09 | 331900 | 0.019643 / 0.010219 | 0.960051 / 3.989% | 0.938185 / 5.820% | 17828 roots, 5.371%, bbox `[829,171]-[1215,288]` |
| 020000 | 32 | 272117 | 0.019927 / 0.010197 | 0.951145 / 4.791% | 0.922238 / 7.382% | 16517 roots, 6.070%, bbox `[738,159]-[961,326]` |
| 030000 | 09 | 331862 | 0.019985 / 0.011599 | 0.955355 / 4.410% | 0.923652 / 7.232% | 17997 roots, 5.423%, bbox `[828,170]-[1181,294]` |
| 030000 | 32 | 272107 | 0.020414 / 0.010608 | 0.944097 / 5.550% | 0.903391 / 9.252% | 18755 roots, 6.893%, bbox `[737,158]-[972,331]` |

Checkpoint-native render reports recorded the following preclip/kept Gaussian
counts for views 09/27:

- 009000: `7197262` preclip; `3763552 / 3349372` kept;
- 020000: `7334346` preclip; `3948715 / 3544177` kept;
- 030000: `7891522` preclip; `4200235 / 3855369` kept.

The low-length basins become materially larger after the 20k geometry unlock
and remain present at 30k in both monitored attribute views. Global opacity
means remain high, while low tip-opacity fractions are higher than low
root-opacity fractions; this supports the baldness concern as a localized
length/support problem rather than a simple global opacity failure.

An independent image-space check uses the checkpoint-native mesh-depth mask as
the visible body support, erodes that support to exclude silhouettes, and then
measures pixels with rendered alpha below `0.5`. At 30k, an 8-pixel erosion
leaves `372291 / 383585` interior pixels in views 09/27. View 09 has zero such
pixels; view 27 has `14` (`0.00365%`), split across four tiny components with a
largest component of nine pixels. With a 16-pixel erosion, both views have
zero. The same 8-pixel check at 20k is zero in view 09 and `10` pixels
(`0.00261%`) in view 27. Therefore the monitored reference views do not contain
large transparent interior holes at 20k or 30k.

This does not erase the attribute warning: overlapping Gaussians can hide
short or individually faint strands in the alpha composite. The concentrated
low-length/low-opacity bands can still appear sparse in pure-strand exports,
PLY inspection, or unmonitored viewpoints. The fixed-view alpha result and the
attribute result are recorded separately rather than collapsing either into a
blanket pass/fail claim.

Per-iteration artifact hash manifests contain:

- 009000: `50` entries (`46` PNG, `4` JSON), manifest SHA-256
  `184e139712d063fb08b7d1310c48a13fb1addacbe63140cd18ea82bc66716c4a`;
- 020000: `50` entries (`46` PNG, `4` JSON), manifest SHA-256
  `1205a19c6bca31c195f4bb1a44b84017e1c8d6a112a799a5822f526c41399121`;
- 030000: `63` entries (`58` PNG, `5` JSON), manifest SHA-256
  `22e3906b2f4189a6ceda9c54fd3899ce0eda48c111ba7d9fec629a10328687c6`.

Coverage execution-log SHA-256 values are:

- `coverage_iter_009000.log`:
  `4d9f3099bca210068cacf5ea49ab7833dc92508cec255acd518ce65c24350245`;
- `coverage_iter_020000.log`:
  `9485b46ef5b4e8253a5da3315b78943136bc3e606d8673e76dae43e551ce01ac`;
- `coverage_iter_030000.log`:
  `fc5f1628c43b2a6684002b81b102913975bae17b42d5b40ad0ced75deaf7c466`;
- corrected post-analysis log:
  `coverage_analysis_retry.log` SHA-256
  `96dbfe73632797367b65497a3bb4fcfc9b6eecb48420670d5bba185fb5e0e127`.

The initial post-analysis attempt is preserved as
`coverage_analysis.log` (SHA-256
`b7aa83098b491d203827e44744301dfe54eb9393fa8f81e7985e72384d29c5ed`); it
failed only because the temporary analyzer omitted the exact source `tools`
path from `sys.path`. The corrected retry completed successfully without
overwriting any coverage artifact.

## Acceptance decision

Terminal training, strict checkpoint validation, final metrics, protected-GPU
checks, sequential 009000/020000/030000 coverage renders, attribute parsing,
connected low-length analysis, and report/image hash recording all completed.
No formal flow/training code was modified, and no qlogin was released or
signaled. The run is accepted as a completed execution artifact; the localized
low-length basins remain baldness-relevant evidence for later method decisions.

## Visual acceptance correction and causal attribution

The completed execution is **not** accepted as a visually valid Panda groom.
Original-resolution user review identifies the dark upper-back length basin as
a real bald patch with coarse surviving hairs and speckled compensation noise.
Calling the basin merely "short hair" was incorrect: the displayed `0.00` is
rounding of a near-zero positive length, not a healthy short-coat value.

Checkpoint-native primary-guide decomposition localizes the failure to the
shared guide field rather than V7 flow or the secondary residual:

| iteration | guide length min / mean | guide width mean / P95 / max | guides <=25% / <=50% of length reference | corr(log length ratio, log width ratio) |
| ---: | ---: | ---: | ---: | ---: |
| 009000 | `0.007274 / 0.010836` | `0.000160 / 0.000160 / 0.000160` | `0 / 0` | `0` |
| 020000 | `0.001914 / 0.020990` | `0.000537 / 0.000941 / 0.001660` | `32 / 142` | `-0.7880` |
| 030000 | `0.001365 / 0.021611` | `0.000539 / 0.001064 / 0.001866` | `74 / 150` | `-0.7374` |

At 30k, the `74` guides at or below one quarter of their own clean-flow-derived
length reference have mean width ratio `9.368x`, median `9.797x`, and minimum
`6.561x`. Their mean length-evidence confidence is `0.746`, above the global
guide mean `0.569`; the optimizer therefore corrupts reliable guide evidence
rather than merely filling an unobserved input region. The secondary-guide
residual changes length only over approximately `0.938x..1.076x` and width over
`0.997x..1.003x`, so it cannot explain either the near-zero length or the
`11.66x` maximum guide-width expansion.

The 30k checkpoint's exact effective root width is also materially coarse:
mean/median/P95/max are
`0.000544 / 0.000490 / 0.001049 / 0.001867`, versus the `0.000160`
reference. The same R068 guide unlock also widens white-tiger fur, but Panda has
a substantially worse coupled short-wide tail: its guide slenderness-expansion
ratio reaches P99/max `44.83x / 96.11x`.

Image-space component isolation agrees with the parameter attribution. In the
reported view-09 bald-patch box `[828,170]-[1181,294]`, shape-detail response is
`4.60x` its mean over the remaining visible body, while Gaussian RGB residual
response is `1.19x`. Removing shape detail exposes a stronger structural hole;
removing Gaussian RGB residual lowers the crop's high-frequency Laplacian
magnitude by about `12%`. The learned shape and appearance outlets therefore
partly conceal the missing support and add visible speckle, but they do not
create the primary failure.

The causal sequence is:

1. primary guides are exact references through 9k;
2. after guide/geometry unlock, coherent guide-length basins collapse while
   guide width expands in the opposite direction;
3. difference-only graph smoothness does not see a coherent regional drift;
4. shape detail and Gaussian RGB residual compensate over the damaged support,
   producing the observed noisy bald patch.

The next isolated candidate is R069, a species-, region-, and view-independent
guide-support gauge. In reference-relative log coordinates it softly penalizes
only length collapse below the stored clean-flow reference and width growth
that exceeds length growth. It uses continuous confidence and intrinsic
surface-area weights plus a population-stable fourth moment. It introduces no
decoded clamp, absolute physical threshold, body mask, view rule, or Panda-only
parameter. Counterfactual rendering and matched Panda/white-tiger short gates
must pass before any new 30k run is authorized.

The completed hard checkpoint counterfactual confirms that correcting only one
coordinate is insufficient. Length restoration alone changes view-09/27
composite PSNR from `30.052/29.544` to `29.726/28.937` but extends the same
over-wide noisy hairs. Enforcing final slenderness alone gives
`25.587/24.184` and exposes a large upper-back alpha hole; the combined hard
projection gives `27.351/26.580`, reduces coarse support, but still exposes the
hole. This is causal evidence for a coupled width-opacity/coverage shortcut,
not a reason to accept a post-hoc clamp. The full diagnostic report is
`D:/RTS/_tmp/panda_r068_v7_acceptance_20260828/counterfactual_support_gauge/counterfactual_support_gauge_report.json`
with SHA-256
`2c008b2400cf388517a5cf27e4201123540605f85fc4dbb44563fc254ce10fa6`.
