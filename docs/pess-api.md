# Structured ESS solver API (schema v1)

## Scope

This API solves the structured **task-wise PESS** and fixed-policy
**Nominal** objectives. It does not implement Joint-Robust. PESS takes
independent task-wise worst cases over all admissible maximal matchings.
Nominal uses one explicitly supplied global candidate priority order; for each
task it scans the selected eligible edges in that order and greedily adds an
edge when it does not conflict with an earlier selected edge. The order is a
known policy input shared across tasks, not an advisor decision or a claim
about PostgreSQL OID order.

The model is deliberately narrower than generic ESS:

- resources are **labeled edges** joining two distinct attributes; parallel
  edges with distinct IDs are allowed;
- a task uses complete **maximal** matchings (not maximum-cardinality
  matchings), with task-specific eligible resources;
- task corrections may have mixed signs and zero-cost resources are allowed;
- task weights must be positive;
- the input loss is `abs(alpha + sum(correction))`, the positive pre-clamp
  absolute-log-error abstraction;
- maintenance is a nonnegative **additive per-resource proxy**. Generic
  `Maint(M)` may be nonadditive and is not implemented by this model.

These restrictions are conditional model assumptions, not claims about every
PostgreSQL query or catalog. Put calibration/loss version, PostgreSQL source
and binary identity, data/workload identity, and cost-model identity in
`input_provenance`; the solver does not infer or alter them.

## Python API

```python
from pess_ess import read_instance, solve

instance = read_instance("instance.json")
result = solve(instance, objective="pess", backend="master_oracle")
nominal = solve(
    instance,
    objective="nominal",
    priority_order=["ab", "bc", "cd"],
)
```

`backend="master_oracle"` is the default and does not enumerate outer resource
subsets. It adds conditional scenario cuts as needed and invokes the task
oracle, which may use Fraction-certified bounds or a floating-point MILP.
`backend="static"` builds the full conditional formulation by
enumerating task matching scenarios; use it only when those families are
tractable. `backend="exact_enumeration"` explicitly enumerates budget-feasible
outer resource sets and uses exact rational arithmetic. Nonnegative additive
costs allow pruning a partial set once its cost exceeds the budget; zero-cost
resources remain eligible. Nominal requires the complete frozen priority order.
The backend returns every exact minimizing set on completion and labels a
timed-out search as interrupted with only its best-known feasible result. It
does not report MILP statuses, bounds, or gaps. The independent
`solve_exhaustive` / `solve_nominal_exhaustive` routines remain the small-instance
differential references. Exact enumeration remains exponential in the worst
case, and does not replace the default MILP backends.

Supported keyword options are `time_limit` (per MILP call), `inner_time_limit`
(per inner task oracle; master-oracle only), `total_time_limit` (global wall
limit), and master-oracle `precompute_invariants` / `task_local_cache`.
Nominal requires `priority_order` to be a complete permutation of all
candidate IDs and uses the `fixed_priority_milp` backend by default. Its
deterministic matching constraints encode the greedy scan without enumerating
resource subsets. Unsupported objective/backend combinations, malformed
priority orders and negative limits raise `ValueError`; priority-order type
errors raise `TypeError`.

The returned `OuterResult.objective` is the Fraction sum of the reported
maximal-matching witness losses. Fraction arithmetic makes that witness sum
exact, but an inner oracle that used HiGHS only has solver-reported numerical
optimality; it does not independently prove that each witness is a true
task-wise worst matching. The compact `result_to_dict(...)` output labels this
as `witness_aggregate`, states whether the selected PESS value is Fraction
certified, and leaves the corresponding global upper bound null otherwise.
An `optimal` status means the floating-point MILP backend reported optimality
and the computed witnesses passed feasibility validation; it is not a
Fraction proof of the outer optimum. `feasible_unproven` means the timed
resource selection and its reported witnesses passed independent feasibility
validation, while global optimality remains unestablished. Inner solver
statuses used by a timeout incumbent are included in `solver_statuses`.
For Nominal, `OuterResult.objective` is the exact Fraction replay of the
specified priority policy on the selected resource set. The MIP still uses
floating-point coefficients: `optimal` is solver-reported optimality, not an
exact proof of the resource-selection optimum. A validated feasible
nonoptimal incumbent is returned as `feasible_unproven`.

## JSON command

Install the package in the environment that will run it, then send one JSON
request on stdin or as a file:

```sh
pess-ess-solve request.json > result.json
```

Request schema v1:

```json
{
  "schema_version": "1",
  "objective": "pess",
  "backend": "master_oracle",
  "limits": {
    "time_seconds": 60,
    "inner_time_seconds": 20,
    "total_time_seconds": 120
  },
  "options": {
    "precompute_invariants": true,
    "task_local_cache": true
  },
  "input_provenance": {
    "model_version": "forest10-positive-preclamp-v1",
    "advisor_source_commit": "<immutable 40-character commit SHA>",
    "research_source_commit": "<immutable 40-character commit SHA>",
    "postgresql_source_commit": "<immutable 40-character commit SHA or null>",
    "calibration_id": "<frozen calibration identifier or null>"
  },
  "instance": {
    "budget": "2",
    "candidates": [
      {"id": "ab", "table": "t", "endpoints": ["a", "b"], "maintenance_cost": "1"}
    ],
    "tasks": [
      {
        "id": "q1",
        "table": "t",
        "weight": "1",
        "alpha": "-1/2",
        "eligible_candidates": ["ab"],
        "corrections": {"ab": "1/2"}
      }
    ]
  }
}
```

For Nominal, use `"objective": "nominal"`, omit `backend` (it defaults to
`fixed_priority_milp`), and add a complete shared order such as
`"priority_order": ["ab", "bc", "cd"]`. The result echoes this policy under
`policy.priority_order`; the selected set remains unordered. Nominal's
`incumbent.objective_value` and Fraction-certified incumbent upper bound are
recomputed by exact rational replay of that policy. The outer optimality
status and lower bound remain floating-point solver reports.

For exact outer enumeration, set `"backend": "exact_enumeration"` and use
`limits.total_time_seconds` as its wall-clock limit. The response reports
`exact_enumeration.complete`, exact `objective_fraction`, all `optimal_sets`
when complete, and the deterministic first set as `canonical_representative`.
When interrupted, it reports `best_known_sets` and makes no global-optimum
claim. `evaluated_subsets` counts complete objective evaluations;
`feasible_subsets` counts feasible leaves reached, including a leaf whose
objective evaluation was interrupted. `pruned_branches` counts budget-infeasible
recursion nodes, not the number of omitted subsets. The existing `static`
backend still means matching-scenario MILP; it is distinct from outer resource
enumeration.

## Add-only greedy backend

Set `"backend": "add_only_greedy"` to use the same strict-improvement
best-addition rule for either objective. It starts from the empty set, evaluates
every remaining budget-feasible candidate with the exact Fraction reference,
and adds the largest strictly positive improvement. Equal improvements use
ascending candidate ID order. It stops at a plateau or when no remaining
candidate is feasible; it does not force budget use. PESS can worsen when a
resource is added, and this backend makes no local- or global-optimum claim.

For Nominal, include the complete fixed global `priority_order`. For PESS,
each fixed-set evaluation maximizes loss over all maximal matchings per task.
An interrupted evaluation is discarded, and the current addition round is
committed only after every feasible candidate in that round has an exact
value. The returned status is `heuristic_completed` or
`heuristic_incomplete`, never `optimal`.

The shared `total_time_seconds` limit applies to both objectives.
`max_evaluations` caps fixed-set evaluations, including the initial empty-set
value. PESS may also use `max_inner_matching_subsets`, a cumulative cap on
matching-subset checks in its exact reference evaluations. The response
separates outer `candidate_attempts`, `fixed_set_evaluation_calls`, completed
evaluations, inner PESS subset checks and maximal matchings scored, and wall
time. Numerical solver limits and Master--Oracle options do not apply to this
backend.

The small synthetic CLI requests are checked in at
[`docs/examples/add-only-greedy-pess.json`](examples/add-only-greedy-pess.json)
and [`docs/examples/add-only-greedy-nominal.json`](examples/add-only-greedy-nominal.json):

```sh
.venv/bin/pess-ess-solve docs/examples/add-only-greedy-pess.json
.venv/bin/pess-ess-solve docs/examples/add-only-greedy-nominal.json
```

Both examples use the same three disjoint synthetic resources and produce the
addition order `a-resource`, `z-resource`, objective `0`, maintenance cost `2`
under budget `3`, and stopping reason `no_strict_improvement`. They are API
smoke examples only; their coefficients are not benchmark or PostgreSQL
measurements.

### Observed synthetic execution

**Experimentally observed, model-only:** Python 3.12.3, `pess-ess` 0.1.0,
implementation commit
[`e454ad431819f2c95289dfbbc79b1572aac8761a`](https://github.com/1951123/ess-advisor/commit/e454ad431819f2c95289dfbbc79b1572aac8761a).
The two checked-in JSON requests above are the complete deterministic inputs;
no random seed or external dataset is used. Both were run with the documented
CLI commands. PESS returned `heuristic_completed`, objective `0`, selected
`[a-resource, z-resource]`, maintenance cost `2`, unused budget `1`, 7 fixed-set
evaluation calls, 23 matching-subset checks and 7 maximal matchings scored. The
Nominal request returned the same selection and objective, with 7 fixed-set
calls and zero inner matching work. The observed wall times were 0.000168 s
(PESS) and 0.000091 s (Nominal); these single-run timings are not a performance
comparison. The plateau test also records a synthetic greedy objective of `1`
versus exact global objective `0` for a two-resource combination; this is a
counterexample to any global-optimality claim, not a performance result.

This smoke run checks API execution and accounting only. It provides no
evidence of Nominal–PESS separation, PostgreSQL behavior, or benchmark
generalization.

Rational values may be JSON strings such as `"-3/2"`, integers, or finite
decimals; they are parsed as exact `Fraction` model inputs. Omit unsupported
`limits` or `options` fields. The result includes `selected`, recomputed
maintenance, the exact Fraction sum of witness losses, feasibility and
optimality labels, solver status codes, scoped solver-reported bounds, wall
time, package/SciPy provenance, and echoed `input_provenance`. A witness sum is
not labeled the true PESS value unless every task maximum used a Fraction
certificate (or the static backend's full matching enumeration). Inner MIP
statuses are floating-point solver reports; if any inner MIP was used, the
global incumbent upper bound and global MIP gap are null. The outer lower
bound, when available, is identified as a solver-reported floating-point
bound; a Master–Oracle bound is from the restricted master at termination and
is not an exact rational certificate.

The JSON response's `phase_timings_seconds` separates Python constraint
construction, row-dictionary/COO assembly, COO-to-CSR conversion, remaining
MILP input preparation (`Bounds`/`LinearConstraint` setup and array conversion),
and the recorded MILP solver-call durations. Per-invocation details are also
included in each `solver_statuses` entry. These are wall-clock diagnostics, not
solver-runtime-only measurements; `wall_time_seconds` remains the overall
request duration.
SciPy's `milp` interface does not expose the bundled HiGHS version, so the
response reports the SciPy version and explicitly leaves `highs_version`
null.

Invalid JSON or unsupported values produce a short stderr message and exit
code 2. A valid request that the numerical solver cannot solve still returns
a JSON result with its solver status and no falsely validated incumbent.

## Reproducible bounded-budget smoke run

The following reuses the frozen six-candidate/four-task model-only instance in
`ess-research@a7aa91e10dd199fd8eb6e5cbaaf1f6e05a607207`,
`experiments/forest10-decision/instance.json` (SHA-256
`afa2f6e21ba35e48352df70ee4287d1a4c71b2fed568cb78ae6d7e24d5929cad`). That
artifact explicitly labels its rational parameters synthetic and not
Forest10/PG measurements. Budget 5 is one of its precommitted budget rows and
is above 3. No research artifact is copied or changed.

```sh
: "${ESS_RESEARCH_ROOT:?Set to a clean checkout at the pinned commit above}"
.venv/bin/python - "$ESS_RESEARCH_ROOT/experiments/forest10-decision/instance.json" <<'PY'
import json
import sys
from pathlib import Path
from pess_ess import solve
from pess_ess.model import instance_from_dict
from pess_ess.reference import solve_exhaustive

source = Path(sys.argv[1])
raw = json.loads(source.read_text(encoding="utf-8"))
raw["budget"] = "5"
instance = instance_from_dict(raw)
result = solve(instance, backend="master_oracle", total_time_limit=10.0)
reference = solve_exhaustive(instance)
assert result.validated_feasible
assert result.objective == reference.objective
print(result.status, sorted(result.selected or ()), result.objective,
      reference.evaluated_subsets, result.total_runtime_seconds)
PY
```

Observed in the Issue #3 Worktree on Python 3.12.3, NumPy 2.5.4, SciPy
1.18.1, NetworkX 3.7, and package 0.1.0: solver status `optimal`, selected
`[e01, e12, e34]`, witness aggregate 14, exact Fraction reference objective
14 with 64 subsets evaluated (44 feasible), 9 master
iterations, 27 distinct cuts, and 0.088823 seconds reported by the solver
(0.088872 seconds around the call). This is one bounded model-level success
and a reference agreement on this six-resource input; it is not a performance
comparison, PG observation, or guarantee for larger instances.

## Immutable consumer pins

The advisor implementation base for this API work is
`1951123/ess-advisor@207cfc41ada05cdc5fa191076749c715fd60c1b9`. The accepted
research semantics were checked against
`1951123/ess-research@a7aa91e10dd199fd8eb6e5cbaaf1f6e05a607207`, including
D-007/D-008 in `docs/DECISIONS.md`. A consumer must record the exact reviewed
and merged advisor commit (or immutable package version) together with schema
version `1`; consumers must not depend on a moving branch or an unmerged
worktree. Research-side data loading and comparisons are a separate task.
