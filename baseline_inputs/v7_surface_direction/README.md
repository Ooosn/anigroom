# V7 Surface Direction Baseline

This directory freezes the accepted white-tiger V7 global-directed-flow target.

Target:

`guide_flow3d_shell_targets_exclude_004_024_025.npz`

- bytes: `4917115`
- SHA-256:
  `f009af820560adf19b6eedbb8bf2c5d29df00cca576be13161b4ee2ebaed6510`
- observed roots: `4407 / 4500`
- global sign changes: `62`
- resolved severe directed edges: `174`
- post-sign ratio updates: `490`
- newly severe directed edges: `0`

Matched Panda target:

- remote/local acceptance filename:
  `guide_flow3d_shell_targets_exclude_004_024_025.npz`
- bytes: `4871812`
- SHA-256:
  `6a220f52b15ca996c88e71802d3309f9499ade442f79dc72300f1af12b5fa56f`
- observed roots: `4194 / 4500`
- global sign changes: `112`
- resolved severe directed edges: `322`
- post-sign ratio updates: `527`
- newly severe directed edges: `0`

Formal generation source:

- global orientation implementation commit: `1664857`;
- exact HGC generation/launcher commit:
  `0712587e2c32c621f5566b7a8706c9dc061fc85b`;
- runtime:
  `/home/wangyy/anigroom-global-directed-v7-20260828`;
- local acceptance:
  `D:/RTS/_tmp/anigroom_v7_formal_results_20260828`.

V7 keeps the V6 robust multiview tangent axis, performs canonical global
binary orientation lifting over trusted surface blocks, then refits only the
nonnegative normal/tangent ratio with a signed directed-edge guard. Panda and
white tiger use the same constants and contain no species, body-region, root,
view-index, or image-coordinate rules.

The complete failure ledger, formulas, fixed-view gate, exact hashes, and
streamline audit are recorded in `docs/v7_global_directed_flow.md`. V6 remains
the immutable fixed-axis parent and V5 remains the completed trained rollback.
