# R084: Topology-Covered RBF Partition-of-Unity Length Field

Status: Phase A pure float64 algebra is implemented and passed. The
mesh-topology cover, checkpoint diagnostic, trainer, configuration, and
visualization paths remain pending or unauthorized; no Panda field evidence
exists.

## Purpose

R083 established exact guide-site identity but failed a real render query: the
query tangent-plane origin lies outside the projected convex hull even when all
`4500` guides are supplied. R084 therefore does not retry the official
natural-neighbor API with a larger ambient candidate set, changed normal, or
tuned parameter. It asks whether a topology-covered collection of exact local
radial basis interpolants can provide a continuous scalar length field while
retaining guide-site identity.

Phase A implements only pure algebra on caller-supplied patch memberships,
radii, nodes, and queries. It does not build the topology cover or evaluate a
checkpoint.

## Decided Contract

1. A later mesh-topology builder supplies patches and all patch memberships.
   Ambient-space proximity never creates patch membership, so nearby folded or
   disconnected sheets cannot exchange guides.
2. Each patch has one fixed topology radius `R`. Radii are mandatory as one
   per-patch tensor of shape `[P]`; a scalar/global broadcast radius is not a
   valid contract. There is no query-adaptive Kth-neighbor radius and no
   truncation that removes a source with nonzero weight.
3. Raw partition-of-unity weights use the fixed-radius Wendland C2 window.
4. For every guide node, each patch active at that node contains that guide in
   its local interpolation support.
5. A local exact RBF uses Euclidean chord distance in `R^3` only after topology
   has certified the patch. Its Wendland C2 kernel scale is `2R`, and its
   augmented polynomial space contains the constant function.
6. Local fields are blended with globally normalized partition-of-unity
   weights. The result is exactly nodal, reproduces constants, and enters or
   leaves a patch through zero raw weight.
7. Exactness does not require the query to lie in a projected convex hull.
   Local/global cardinal coefficients may be signed; no convex-hull or convex-
   combination guarantee is part of R084.
8. Singular algebra, uncovered queries, invalid topology membership, duplicate
   nodes, nonfinite values, or a zero PU denominator fail explicitly. There is
   no fallback, diagonal jitter, pseudoinverse, least-squares solve,
   regularization, or silent patch reduction. The implementation has no NumPy
   or SciPy path and uses no `pinv` or `lstsq`.
9. R084 handles scalar physical length only. Direction remains a separate
   normal-compatible parallel-transport problem.
10. Any guide lifecycle change must rebuild patch memberships and local patch
    algebra. Continuous render-root barycentric movement may be evaluated later
    against fixed valid patch algebra; that runtime path is not yet authorized.
11. Every solved local system is bound to its patch identity. Evaluation must
    verify exact system-to-patch identity and ordering; a system from another
    patch may not be substituted even when tensor shapes match.

## Algebra

Let patch `a` have fixed topology radius `R_a`, topology-certified guide IDs
`I_a`, and a topology distance or membership coordinate `d_a(x)` supplied by
the later cover builder. The fixed Wendland C2 window is

```text
phi(q) = (1 - q)^4 (4q + 1),  0 <= q < 1
phi(q) = 0,                    q >= 1.
```

The raw PU weight is

```text
omega_a(x) = phi(d_a(x) / R_a).
```

No query changes `R_a`. A patch transition occurs only where `omega_a` reaches
zero.

Within topology-certified patch `a`, define the local RBF matrix from 3D chord
distance at scale `2R_a`:

```text
Phi_a[i,j] = phi(||p_i - p_j||_2 / (2 R_a)),  i,j in I_a.
```

For scalar guide lengths `ell_i`, solve the exact constant-augmented system

```text
[ Phi_a   1 ] [lambda_a] = [ell_a]
[   1^T   0 ] [   c_a  ]   [  0  ].
```

The local interpolant is

```text
s_a(x) = sum_{j in I_a} lambda_a[j]
         * phi(||x - p_j||_2 / (2 R_a)) + c_a.
```

The global field is the normalized blend over active patches:

```text
s(x) = sum_a omega_a(x) s_a(x) / sum_a omega_a(x).
```

At guide node `p_i`, every active patch contains `i` and therefore evaluates
to `ell_i`; the normalized blend is also exactly `ell_i`. Constant polynomial
augmentation makes every local patch reproduce a constant, so the global
normalized blend reproduces the same constant. Compact raw PU weights make
patch entry/exit occur at zero contribution rather than by replacing a
nonzero truncated neighbor.

## Phase A Result: Pure Algebra Passed

The predeclared float64 algebra gates all pass:

1. maximum node self-evaluation error `<= 1e-10`;
2. maximum constant-field reproduction error `<= 1e-10`;
3. maximum global cardinal-function sum error `<= 1e-10`;
4. finite query-position autograd;
5. continuous values across patch support boundaries;
6. a folded/disconnected topology fixture with no cross-sheet membership;
7. strict failure for singular local systems and uncovered queries; and
8. the focused R084 tests and complete repository test suite pass.

Phase A implementation is isolated to:

- `anigroom/rbf_partition_of_unity.py`;
- `tests/test_rbf_partition_of_unity.py`.

The implementation requires a per-patch radius tensor of shape `[P]`, binds
every solved system to its exact patch identity, and contains no NumPy, SciPy,
`pinv`, `lstsq`, diagonal jitter, or regularized fallback path.

Validation evidence:

- `14` focused tests passed in `1.78 s`;
- full `mygs` pytest: `578` passed, `14` warnings in `19.56 s`;
- `py_compile` passed; and
- `git diff --check` passed.

Phase A success authorizes neither a mesh-topology cover implementation nor a
fixed-checkpoint run. The later cover and checkpoint gates are explicitly not
yet defined or authorized, and no actual Panda field evidence is claimed.

## Topology And Lifecycle Boundary

The future topology builder owns patch centers, fixed radii, memberships,
coverage proof, and folded/disconnected-surface exclusion. The algebra layer
must not infer missing membership with ambient KNN, expand a radius after seeing
a query, or pad an undersupported patch.

Guide insertion, deletion, or movement changes interpolation nodes and must
rebuild affected patch systems. Render-root barycentric movement does not
change guide algebra and can in principle be evaluated continuously while its
certified patch cover remains valid, but that evaluation contract belongs to a
later gate.

## Constrained Harmonic FEM Audit

A read-only audit considered constrained harmonic finite elements as a
mathematically valid alternative. The Panda mesh has `340288` vertices and
`680572` faces. The audited constrained KKT system would have dimension
`344788` with approximately `2.409 million` nonzeros; the mesh has `173294`
negative cotangent edges. The current environment provides no CHOLMOD, libigl,
or geometry-central backend, and the model would require `9-18` right-hand
sides per forward evaluation.

This does not reject harmonic interpolation mathematically. It means a current
full-mesh constrained solve on every forward fails the R084 efficiency boundary
and was not selected as the training-time route. No harmonic candidate was
implemented or executed, so this audit is not an experimental rejection.

## Mature Foundations

- [Wendland (1998), "Error Estimates for Interpolation by Compactly Supported
  Radial Basis Functions of Minimal Degree"](https://doi.org/10.1006/jath.1997.3137)
  provides the compactly supported positive-definite RBF foundation.
- [Partition of unity interpolation using stable kernel-based
  techniques](https://arxiv.org/abs/1607.03278) records the mature RBF-PU
  principle of local kernel approximants blended by locally supported PU
  weights and discusses stable local evaluation.
- [Dynamic Harmonic Fields for Surface
  Processing](https://doi.org/10.1016/j.cag.2009.03.022) is the considered
  harmonic-field alternative; it is cited as context, not as an executed R084
  candidate.
- The official [geometry-central intrinsic triangulation
  documentation](https://geometry-central.net/surface/intrinsic_triangulations/basics/)
  documents mature intrinsic-triangulation and original-mesh correspondence
  machinery. R084 does not currently depend on geometry-central.

## Decision Rule

R084 Phase A pure algebra is complete and passed. This does not authorize
topology construction, checkpoint evaluation, trainer/config integration, or
visualization, and it is not actual Panda field evidence. Any later gate must
retain exact nodal identity, constant/cardinal reproduction, finite gradients,
support-boundary continuity, folded-topology isolation, strict
singular/uncovered failure, per-patch `[P]` radii, and system-to-patch identity
binding.
