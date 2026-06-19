# anigroom

Editable animal grooming from mesh-rooted parametric fur.

## White Tiger Mainline

The active route is intentionally small:

- Stage 1 trains mesh-rooted fur grooming parameters.
- Stage 1 may use random colored mesh backing only to prevent white-fur transparency during reconstruction training.
- Stage 2 is not part of the current active run.
- UV maps are storage and visualization; geometric smoothness should be enforced on mesh/root neighborhoods.

See `docs/MAINLINE.md` for the current entry points.
