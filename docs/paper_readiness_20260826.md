# Paper Readiness Addendum: R068

Status date: 2026-08-26.

R068 replaces R067 as the current single-sample method baseline. It does not
change the paper's representation or evidence claims: it removes crossing from
default training and adds an exact frozen-phase runtime path. See
`docs/r068_no_crossing_zero_curl.md` for the complete gate.

R068 improves the accepted white-tiger evidence:

- wall time `15775.028 -> 11885.196 s` (`-24.66%`);
- peak allocation `19825.54 -> 15869.80 MB` (`-19.95%`);
- fixed eight-view composite `33.077009 -> 33.101055`;
- backward/foldback remain zero and matched canonical assets remain coherent;
- no-penetration slightly improves;
- crossing's small offline diagnostic benefit is retained as an ablation, not
  paid in the default objective.

The publication gaps identified in `docs/paper_readiness_20260825.md` remain:
multi-scene and multi-seed evidence, comparable external baselines, trained
checkpoint edit demonstrations, and compact final ablation/lifecycle figures.
R068 improves method efficiency and cleans the default objective; it does not
claim those missing experiments are complete.

Accepted local evidence:

`D:/RTS/_tmp/r068_acceptance_20260826/postprocess/r068_no_crossing_zero_curl`
