# R083: Surface Natural-Neighbor Length Field

Status: official dependency/bootstrap and folded-surface API gate passed;
local weight-builder/I/O implementation complete; HGC cross-language compile
gate pending; no training or visualization authorized.

## Purpose

R081 and R082 show that normalized compact kernels cannot satisfy both field
continuity and guide-root attribute semantics. R081 remains sharply segmented;
R082 improves global edge statistics by averaging guide values so strongly that
guide-site relative error reaches `17.163%` at P95 and `72.691%` maximum.

R083 replaces the kernel family rather than changing another support radius.
It tests surface Natural Neighbor/Sibson coordinates, the standard
Voronoi-derived nodal coordinates that match the user's intended root roles:

- each primary guide root is an attribute node;
- guide-root density defines local Voronoi cell size and therefore spatial
  resolution;
- weights are nonnegative and form a partition of unity;
- the Kronecker property preserves each guide's value at its own site;
- neighboring sites enter and leave through continuous natural-neighbor
  coordinates rather than a truncated KNN slot.

## Reviewed External Implementation

R083 uses the official CGAL 6.2
`CGAL::surface_neighbor_coordinates_3()` implementation. It does not recreate
Sibson weights in Python or replace the official routine with a local heuristic.

Pinned isolated dependencies on HGC:

- CGAL 6.2 full source archive SHA-256:
  `fbc32816745e871a5cbdeb6245317e9dbf10ae1a957b0ab1edb00b4fde00ba8d`;
- Boost 1.86.0 source archive SHA-256:
  `1bed88e40401b2cb7a1f76d4bab499e352fa4d0c5f31c0dbae64e24d34d7513b`;
- system GMP/MPFR; no system package, conda environment, or AniGroom source was
  modified during bootstrap.

The unmodified CGAL
`examples/Interpolation/surface_neighbor_coordinates_3.cpp` compiles under the
HGC GCC 13.2 toolchain with no warning and exits zero. Bootstrap evidence is
under:

`D:/RTS/_tmp/panda_r083_natural_neighbor_bootstrap_20260901`.

## Official API Route

Build one reusable `CGAL::Delaunay_triangulation_3` from the fixed primary
guide sites. For each render-root query and its surface normal, call the
official Delaunay-filtered overload:

```cpp
CGAL::surface_neighbor_coordinates_3(
    delaunay,
    query_point,
    query_normal,
    output_iterator);
```

The official wrapper locates the query in the reusable 3D Delaunay
triangulation, extracts conflict-boundary candidates, and creates the local 2D
regular triangulation in the query tangent plane. No manual candidate filter is
allowed in the R083 method gate.

The wrapper returns source points rather than source handles. R083 maps each
returned point to its original guide ID through an exact deterministic
point-key map and fails on duplicate guide coordinates or an unmatched returned
point.

## Folded/Disconnected-Surface Probe

Before project implementation, an isolated official-API probe used a 5x5 grid
on one sheet and matching disconnected sheets at separations
`0.1x, 0.5x, 1x, 2x, 5x` grid spacing. Every case returns the same four
same-sheet natural neighbors:

```text
opposite-sheet total weight = 0
maximum opposite-sheet weight = 0
weight sum = 1
barycentric reconstruction error = 0
```

The Delaunay-filtered and all-points range overloads return identical IDs and
weights. Probe result SHA-256:
`227731c3dea6f097bb895af62968c7dbcf8304029374635eea9647885a8f562c`.
Evidence is under:

`D:/RTS/_tmp/panda_r083_natural_neighbor_bootstrap_20260901/folded_sheet_probe`.

This probe does not by itself prove arbitrary animal topology. The complete
Panda and white-tiger gates must additionally audit every returned guide
against the existing intrinsic topology-safe support and stop on a cross-sheet
or uncertified query.

## Phase A: Weight Builder Contract

R083 first implements only a deterministic standalone C++ weight builder and a
Python binary I/O validator.

Input:

- fixed guide points with stable IDs;
- render-root query points;
- finite normalized query surface normals.

Output is a versioned little-endian CSR matrix:

- row offsets for every query;
- original guide IDs;
- normalized natural-neighbor weights;
- per-query success and barycentric reconstruction evidence.

The builder must:

1. build the guide Delaunay triangulation once;
2. call only the official surface-neighbor coordinate overload;
3. fail on undersampling, nonfinite/negative weights, zero normalization,
   duplicate IDs, unmatched points, invalid normals, or nonfinite
   reconstruction evidence;
4. sort each output row by guide ID for deterministic bytes;
5. write atomically and preserve complete provenance;
6. expose no training, rendering, or checkpoint mutation path.

No local CGAL dependency is required for Python I/O tests. The C++ builder is
compiled and tested only against the pinned HGC bootstrap until a reviewed
portable dependency contract exists.

Local implementation files:

- `tools/cpp/surface_natural_neighbor_weights.cpp`;
- `tools/surface_natural_neighbor_io.py`;
- `tests/test_surface_natural_neighbor_io.py`.

The binary contract uses strict versioned little-endian float64 inputs and CSR
outputs. The C++ producer refuses an existing destination, reserves an
exclusive temporary sibling directory, verifies final byte count, and publishes
only after all official CGAL queries pass. Query normals must already be unit
length within `1e-5`, which admits normalized float32 model normals without
silently changing their direction.

Local Python verification reports `44` focused tests and `536` complete
repository tests passing. Local validation does not claim C++ compatibility;
the exact source must still compile on HGC against the pinned CGAL/Boost tree
and pass a Python-writer -> C++-builder -> Python-reader golden roundtrip.

## Phase B: Fixed-Checkpoint Gates

Before the complete Panda population:

1. official sphere example;
2. single-plane and disconnected folded-sheet probes;
3. guide-site queries proving Kronecker identity;
4. deterministic bounded Panda subsets for correctness and timing;
5. complete Panda R080 iteration-4000 weights only after subset acceptance.

Complete Panda diagnostics then use the exact inherited surface edge set and
record:

- guide-site self error;
- constant/linear reproduction and partition of unity;
- natural-neighbor count distribution;
- field difference from inherited K8;
- complete edge log-length statistics;
- natural-neighbor-set changed/unchanged partitions;
- intrinsic-support containment and no-cross-sheet audit;
- build/evaluation time, CSR bytes, and peak memory.

Panda must pass before white tiger. Both must pass before the unchanged
canonical `tools/visualize_white_tiger_groom_attributes.py` is allowed to
render the field. Trainer/config/checkpoint integration is a separate later
gate.

## Efficiency Boundary

Natural-neighbor weights are constructed offline when guide/query topology is
rebuilt. Training-time field evaluation, if eventually accepted, is one sparse
CSR gather/reduction; it contains no per-step Voronoi construction or global
linear solve.

The current CGAL wrapper still builds a small regular triangulation for each
query after reusable Delaunay filtering. R083 therefore benchmarks bounded
subsets before authorizing 496,632 queries. Parallel shared-read queries are not
assumed thread-safe without separate evidence.

## Literature And Official Documentation

- https://doc.cgal.org/latest/Interpolation/index.html
- https://doi.org/10.1016/S0925-7721(01)00018-9
- https://doi.org/10.1002/1097-0207(20010110)50:1%3C1::AID-NME14%3E3.0.CO;2-P

## Decision Rule

R083 remains diagnostic. Any undersampling, cross-sheet contribution, loss of
nodal identity, excessive runtime, nondeterministic CSR output, or need for a
manual neighborhood heuristic rejects the candidate before training.
