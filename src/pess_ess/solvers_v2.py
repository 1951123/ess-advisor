"""Validated static and Master--Oracle outer solvers.

The implementation keeps the original mathematical formulations, but exposes
solver bounds, numerical-status qualifiers, global deadlines, and phase
diagnostics needed by reproducible benchmarks.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction

from .matchings import enumerate_matchings
from .mip import MIPResult, solve_binary_milp
from .model import Instance, Task
from .oracle import InnerOracleResult, evaluate_task, ordinary_matching_extrema
from .reference import reference_objective
from .validation import validate_outer_solution


@dataclass(frozen=True)
class OuterResult:
    status: str
    termination_reason: str
    selected: frozenset[str] | None
    maintenance_cost: Fraction | None
    objective: Fraction | None
    lower_bound: float | None
    upper_bound: float | None
    task_details: dict[str, tuple[Fraction, frozenset[str]]]
    master_iterations: int
    distinct_cuts: int
    inner_oracle_calls: int
    certified_early_exits: int
    scenario_count: int
    solver_statuses: tuple[MIPResult, ...]
    solver_reported_optimal: bool = False
    validated_feasible: bool = False
    reference_agreement: bool | None = None
    master_primal_bound: float | None = None
    master_dual_bound: float | None = None
    master_mip_gap: float | None = None
    numerical_warnings: tuple[str, ...] = ()
    numerical_violation_count: int = 0
    existing_cut_violation_count: int = 0
    task_evaluations: int = 0
    ordinary_matching_bound_evaluations: int = 0
    exact_oracle_invocations: int = 0
    inner_mip_solve_calls: int = 0
    highs_invocations: int = 0
    presolve_fallback_attempts: int = 0
    presolve_fallback_successes: int = 0
    master_phase_seconds: float = 0.0
    ordinary_matching_seconds: float = 0.0
    exact_inner_seconds: float = 0.0
    cut_construction_seconds: float = 0.0
    validation_seconds: float = 0.0
    total_runtime_seconds: float = 0.0
    invariant_precomputation_seconds: float = 0.0
    cache_management_seconds: float = 0.0
    task_evaluation_requests: int = 0
    task_evaluation_computations: int = 0
    distinct_task_local_configurations: int = 0
    cache_lookups: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_entries: int = 0
    cache_hit_rate: float = 0.0
    full_graph_ordinary_matching_extrema_computations: int = 0
    task_local_ordinary_matching_bound_computations: int = 0


@dataclass(frozen=True)
class MasterOracleOptions:
    """Independent switches for the two Master--Oracle optimizations."""

    precompute_invariants: bool = False
    task_local_cache: bool = False


class TaskLocalOracleCache:
    """Invocation-scoped cache for completed task-local evaluations."""

    def __init__(self) -> None:
        self._values: dict[
            tuple[str, frozenset[str]], InnerOracleResult
        ] = {}

    @staticmethod
    def key(task: Task, selected: frozenset[str]) -> tuple[str, frozenset[str]]:
        local = frozenset(selected & set(task.eligible_candidates))
        return task.id, local

    @staticmethod
    def cacheable(result: InnerOracleResult) -> bool:
        if result.source == "certified_bounds":
            return result.lower_bound == result.upper_bound
        return result.source == "mip" and all(
            status.optimal for status in result.solver_statuses
        )

    def get(
        self, task: Task, selected: frozenset[str]
    ) -> InnerOracleResult | None:
        return self._values.get(self.key(task, selected))

    def put(
        self,
        task: Task,
        selected: frozenset[str],
        result: InnerOracleResult,
    ) -> bool:
        if not self.cacheable(result):
            return False
        self._values[self.key(task, selected)] = result
        return True

    def __len__(self) -> int:
        return len(self._values)


def _floor(instance: Instance, task: Task) -> Fraction:
    extrema = ordinary_matching_extrema(instance, task)
    return max(
        Fraction(),
        task.alpha + extrema.minimum_value,
        -task.alpha - extrema.maximum_value,
    )


def _cut_row(
    instance: Instance,
    task: Task,
    matching: frozenset[str],
    lambdas: dict[str, Fraction],
    x_index: dict[str, int],
    z_index: int,
) -> tuple[dict[int, float], float, float]:
    candidates = instance.candidate_map
    occupied = (
        set().union(*(set(candidates[item].endpoints) for item in matching))
        if matching
        else set()
    )
    disjoint = frozenset(
        candidate_id
        for candidate_id in task.eligible_candidates
        if not (set(candidates[candidate_id].endpoints) & occupied)
    )
    loss = abs(
        task.alpha + sum((task.corrections[item] for item in matching), Fraction())
    )
    coefficient = loss - lambdas[task.id]
    row: dict[int, float] = {z_index: 1.0}
    for candidate_id in matching:
        row[x_index[candidate_id]] = row.get(x_index[candidate_id], 0.0) - float(
            coefficient
        )
    for candidate_id in disjoint:
        row[x_index[candidate_id]] = row.get(x_index[candidate_id], 0.0) + float(
            coefficient
        )
    constant = loss - coefficient * len(matching)
    return row, float(constant), float("inf")


def _solve_conditional_master(
    instance: Instance,
    cuts: set[tuple[str, tuple[str, ...]]],
    time_limit: float | None,
    *,
    lambdas: dict[str, Fraction] | None = None,
    deadline: float | None,
    clock: Callable[[], float],
) -> tuple[MIPResult, dict[str, Fraction], float]:
    build_started = clock()
    candidate_ids = instance.candidate_ids
    task_ids = tuple(task.id for task in instance.tasks)
    x_index = {candidate_id: index for index, candidate_id in enumerate(candidate_ids)}
    z_index = {
        task_id: len(candidate_ids) + index for index, task_id in enumerate(task_ids)
    }
    if lambdas is None:
        lambdas = {task.id: _floor(instance, task) for task in instance.tasks}
    objective = [0.0] * len(candidate_ids) + [
        float(task.weight) for task in instance.tasks
    ]
    integrality = [1] * len(candidate_ids) + [0] * len(task_ids)
    lower_bounds = [0.0] * len(candidate_ids) + [
        float(lambdas[task.id]) for task in instance.tasks
    ]
    upper_bounds = [1.0] * len(candidate_ids) + [float("inf")] * len(task_ids)
    rows: list[tuple[dict[int, float], float, float]] = [
        (
            {
                x_index[candidate.id]: float(candidate.maintenance_cost)
                for candidate in instance.candidates
            },
            -float("inf"),
            float(instance.budget),
        )
    ]
    for task_id, matching_tuple in sorted(cuts):
        task = instance.task_map[task_id]
        rows.append(
            _cut_row(
                instance,
                task,
                frozenset(matching_tuple),
                lambdas,
                x_index,
                z_index[task_id],
            )
        )
    build_elapsed = clock() - build_started
    result = solve_binary_milp(
        objective=objective,
        integrality=integrality,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        rows=rows,
        time_limit=time_limit,
        deadline=deadline,
        clock=clock,
    )
    return result, lambdas, build_elapsed


def _extract_selection(instance: Instance, result: MIPResult) -> frozenset[str]:
    if len(result.values) < len(instance.candidates):
        raise RuntimeError("MIP did not return a complete selection vector")
    selected = set()
    for index, candidate in enumerate(instance.candidates):
        value = result.values[index]
        if abs(value - round(value)) > 1e-5:
            raise RuntimeError("MIP returned a fractional integer selection")
        if round(value):
            selected.add(candidate.id)
    return frozenset(selected)


def _master_lower_bound(result: MIPResult) -> float | None:
    if result.dual_bound is not None:
        return result.dual_bound
    return result.primal_objective if result.optimal else None


def _empty_result(
    *,
    status: str,
    reason: str,
    iterations: int,
    cuts: int,
    scenarios: int,
    statuses: tuple[MIPResult, ...],
    selected: frozenset[str] | None = None,
    objective: Fraction | None = None,
    details: dict[str, tuple[Fraction, frozenset[str]]] | None = None,
    instance: Instance | None = None,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    **metrics,
) -> OuterResult:
    extras = {
        key: value
        for key, value in metrics.items()
        if key in OuterResult.__dataclass_fields__
    }
    certified = extras.pop(
        "certified_early_exits",
        metrics.get("certified_early_exits", 0),
    )
    exact_oracles = extras.get(
        "exact_oracle_invocations",
        metrics.get("exact_oracle_invocations", 0),
    )
    return OuterResult(
        status=status,
        termination_reason=reason,
        selected=selected,
        maintenance_cost=(
            instance.maintenance_cost(selected)
            if instance is not None and selected is not None
            else None
        ),
        objective=objective,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        task_details=details or {},
        master_iterations=iterations,
        distinct_cuts=cuts,
        inner_oracle_calls=exact_oracles,
        certified_early_exits=certified,
        scenario_count=scenarios,
        solver_statuses=statuses,
        **extras,
    )


def solve_static(
    instance: Instance,
    time_limit: float | None = None,
    *,
    total_time_limit: float | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> OuterResult:
    started = clock()
    deadline = started + total_time_limit if total_time_limit is not None else None
    all_matchings = []
    cuts: set[tuple[str, tuple[str, ...]]] = set()
    for task in instance.tasks:
        endpoints = {
            candidate.id: candidate.endpoints
            for candidate in instance.task_candidates(task)
        }
        matchings = enumerate_matchings(task.eligible_candidates, endpoints)
        all_matchings.append((task.id, matchings))
        cuts.update((task.id, tuple(sorted(matching))) for matching in matchings)
    result, _, build_seconds = _solve_conditional_master(
        instance,
        cuts,
        time_limit,
        deadline=deadline,
        clock=clock,
    )
    master_seconds = result.total_duration_seconds
    scenario_count = sum(len(matchings) for _, matchings in all_matchings)
    if not result.optimal:
        return _empty_result(
            status="not_optimal",
            reason=result.message,
            iterations=1,
            cuts=len(cuts),
            scenarios=scenario_count,
            statuses=(result,),
            instance=instance,
            lower_bound=_master_lower_bound(result),
            master_primal_bound=result.primal_objective,
            master_dual_bound=result.dual_bound,
            master_mip_gap=result.mip_gap,
            solver_reported_optimal=False,
            validated_feasible=False,
            master_phase_seconds=master_seconds,
            cut_construction_seconds=build_seconds,
            total_runtime_seconds=clock() - started,
            highs_invocations=result.highs_invocations,
            presolve_fallback_attempts=int(result.presolve_fallback_attempted),
            presolve_fallback_successes=result.presolve_fallback_successes,
        )
    selected = _extract_selection(instance, result)
    validation_started = clock()
    objective, details = reference_objective(instance, selected)
    validation = validate_outer_solution(instance, selected, details)
    validation_seconds = clock() - validation_started
    status = "optimal" if validation.valid else "unresolved_validation"
    reason = (
        "static full conditional formulation solved"
        if validation.valid
        else "; ".join(validation.errors)
    )
    return _empty_result(
        status=status,
        reason=reason,
        iterations=1,
        cuts=len(cuts),
        scenarios=scenario_count,
        statuses=(result,),
        selected=selected,
        objective=objective if validation.valid else None,
        details=details if validation.valid else {},
        instance=instance,
        lower_bound=_master_lower_bound(result),
        upper_bound=float(objective) if validation.valid else None,
        solver_reported_optimal=True,
        validated_feasible=validation.valid,
        master_primal_bound=result.primal_objective,
        master_dual_bound=result.dual_bound,
        master_mip_gap=result.mip_gap,
        master_phase_seconds=master_seconds,
        cut_construction_seconds=build_seconds,
        validation_seconds=validation_seconds,
        total_runtime_seconds=clock() - started,
        highs_invocations=result.highs_invocations,
        presolve_fallback_attempts=int(result.presolve_fallback_attempted),
        presolve_fallback_successes=result.presolve_fallback_successes,
    )


def solve_master_oracle(
    instance: Instance,
    time_limit: float | None = None,
    inner_time_limit: float | None = None,
    certification_tolerance: float = 0.0,
    *,
    cut_tolerance: float = 1e-6,
    total_time_limit: float | None = None,
    options: MasterOracleOptions | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> OuterResult:
    options = options or MasterOracleOptions()
    started = clock()
    deadline = started + total_time_limit if total_time_limit is not None else None
    invariant_started = clock()
    invariant_lambdas = (
        {task.id: _floor(instance, task) for task in instance.tasks}
        if options.precompute_invariants
        else None
    )
    invariant_seconds = clock() - invariant_started
    full_graph_extrema_computations = (
        len(instance.tasks) if options.precompute_invariants else 0
    )
    task_cache = TaskLocalOracleCache() if options.task_local_cache else None
    cuts: set[tuple[str, tuple[str, ...]]] = set()
    solver_statuses: list[MIPResult] = []
    best_selection: frozenset[str] | None = None
    best_objective: Fraction | None = None
    best_details: dict[str, tuple[Fraction, frozenset[str]]] = {}
    lower_bound: float | None = None
    certified_exits = 0
    task_evaluations = 0
    exact_oracles = 0
    inner_mip_calls = 0
    highs_calls = 0
    fallback_attempts = 0
    fallback_successes = 0
    ordinary_seconds = 0.0
    exact_seconds = 0.0
    cut_seconds = 0.0
    master_seconds = 0.0
    iterations = 0
    numerical_warnings: list[str] = []
    numerical_violations = 0
    existing_cut_violations = 0
    task_evaluation_requests = 0
    task_evaluation_computations = 0
    distinct_task_local_configurations: set[tuple[str, frozenset[str]]] = set()
    cache_lookups = 0
    cache_hits = 0
    cache_misses = 0
    cache_management_seconds = 0.0
    task_local_bound_computations = 0

    def metrics() -> dict[str, object]:
        return {
            "certified_early_exits": certified_exits,
            "task_evaluations": task_evaluations,
            "ordinary_matching_bound_evaluations": task_local_bound_computations,
            "exact_oracle_invocations": exact_oracles,
            "inner_mip_solve_calls": inner_mip_calls,
            "highs_invocations": highs_calls,
            "presolve_fallback_attempts": fallback_attempts,
            "presolve_fallback_successes": fallback_successes,
            "master_phase_seconds": master_seconds,
            "ordinary_matching_seconds": ordinary_seconds,
            "exact_inner_seconds": exact_seconds,
            "cut_construction_seconds": cut_seconds,
            "total_runtime_seconds": clock() - started,
            "numerical_warnings": tuple(numerical_warnings),
            "numerical_violation_count": numerical_violations,
            "existing_cut_violation_count": existing_cut_violations,
            "invariant_precomputation_seconds": invariant_seconds,
            "cache_management_seconds": cache_management_seconds,
            "task_evaluation_requests": task_evaluation_requests,
            "task_evaluation_computations": task_evaluation_computations,
            "distinct_task_local_configurations": len(
                distinct_task_local_configurations
            ),
            "cache_lookups": cache_lookups,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "cache_entries": len(task_cache) if task_cache is not None else 0,
            "cache_hit_rate": (
                cache_hits / cache_lookups if cache_lookups else 0.0
            ),
            "full_graph_ordinary_matching_extrema_computations": (
                full_graph_extrema_computations
            ),
            "task_local_ordinary_matching_bound_computations": (
                task_local_bound_computations
            ),
        }

    while True:
        iterations += 1
        if deadline is not None and clock() >= deadline:
            return _empty_result(
                status="not_optimal",
                reason="global deadline exhausted before master solve",
                iterations=iterations,
                cuts=len(cuts),
                scenarios=len(cuts),
                statuses=tuple(solver_statuses),
                selected=best_selection,
                objective=best_objective,
                details=best_details,
                instance=instance,
                lower_bound=lower_bound,
                upper_bound=float(best_objective)
                if best_objective is not None
                else None,
                solver_reported_optimal=False,
                validated_feasible=best_selection is not None,
                **metrics(),
            )
        if invariant_lambdas is None:
            full_graph_extrema_computations += len(instance.tasks)
        result, _lambdas, build_seconds = _solve_conditional_master(
            instance,
            cuts,
            time_limit,
            lambdas=invariant_lambdas,
            deadline=deadline,
            clock=clock,
        )
        solver_statuses.append(result)
        master_seconds += result.total_duration_seconds
        cut_seconds += build_seconds
        lower_bound = _master_lower_bound(result)
        highs_calls += result.highs_invocations
        fallback_attempts += int(result.presolve_fallback_attempted)
        fallback_successes += result.presolve_fallback_successes
        if not result.optimal:
            validation = (
                validate_outer_solution(instance, best_selection, best_details)
                if best_selection is not None
                else None
            )
            return _empty_result(
                status="not_optimal",
                reason=result.message,
                iterations=iterations,
                cuts=len(cuts),
                scenarios=len(cuts),
                statuses=tuple(solver_statuses),
                selected=best_selection,
                objective=best_objective,
                details=best_details,
                instance=instance,
                lower_bound=lower_bound,
                upper_bound=float(best_objective)
                if best_objective is not None
                else None,
                solver_reported_optimal=False,
                validated_feasible=validation.valid if validation else False,
                master_primal_bound=result.primal_objective,
                master_dual_bound=result.dual_bound,
                master_mip_gap=result.mip_gap,
                **metrics(),
            )
        selected = _extract_selection(instance, result)
        changed = False
        details: dict[str, tuple[Fraction, frozenset[str]]] = {}
        task_index = {task.id: index for index, task in enumerate(instance.tasks)}
        for task in instance.tasks:
            task_evaluation_requests += 1
            task_evaluations += 1
            cache_started = clock()
            local_key = TaskLocalOracleCache.key(task, selected)
            distinct_task_local_configurations.add(local_key)
            cached = None
            if task_cache is not None:
                cache_lookups += 1
                cached = task_cache.get(task, selected)
                if cached is None:
                    cache_misses += 1
                else:
                    cache_hits += 1
            cache_management_seconds += clock() - cache_started
            if cached is not None:
                evaluation = cached
            else:
                evaluation = evaluate_task(
                    instance,
                    task,
                    selected,
                    time_limit=inner_time_limit,
                    certification_tolerance=certification_tolerance,
                    reuse_bounds=options.precompute_invariants,
                    deadline=deadline,
                    clock=clock,
                )
                task_evaluation_computations += 1
                task_local_bound_computations += (
                    evaluation.ordinary_matching_bound_computations
                )
                if task_cache is not None:
                    cache_started = clock()
                    task_cache.put(task, selected, evaluation)
                    cache_management_seconds += clock() - cache_started
            if cached is None:
                exact_oracles += evaluation.exact_oracle_invocations
                inner_mip_calls += evaluation.inner_mip_solve_calls
                certified_exits += int(evaluation.certified_early_exit)
                ordinary_seconds += evaluation.ordinary_matching_seconds
                exact_seconds += evaluation.exact_inner_seconds
                highs_calls += evaluation.highs_invocations
                fallback_attempts += evaluation.presolve_fallback_attempts
                fallback_successes += evaluation.presolve_fallback_successes
            if evaluation.source == "mip_nonoptimal":
                return _empty_result(
                    status="not_optimal",
                    reason="inner MIP did not prove optimality",
                    iterations=iterations,
                    cuts=len(cuts),
                    scenarios=len(cuts),
                    statuses=tuple(solver_statuses) + evaluation.solver_statuses,
                    selected=selected,
                    instance=instance,
                    lower_bound=lower_bound,
                    solver_reported_optimal=True,
                    validated_feasible=False,
                    **metrics(),
                )
            if cached is None:
                solver_statuses.extend(evaluation.solver_statuses)
            details[task.id] = (evaluation.value, evaluation.worst_matching)
            z_value = result.values[len(instance.candidates) + task_index[task.id]]
            violation = float(evaluation.value) - z_value
            if violation > 0.0:
                if violation <= cut_tolerance:
                    numerical_warnings.append(
                        f"{task.id}: positive cut violation {violation:.3g} "
                        f"within tolerance {cut_tolerance:.3g}"
                    )
                    numerical_violations += 1
                    continue
                cut_key = (task.id, tuple(sorted(evaluation.worst_matching)))
                if cut_key in cuts:
                    existing_cut_violations += 1
                    numerical_warnings.append(
                        f"{task.id}: violated cut already present ({violation:.3g})"
                    )
                    validation = validate_outer_solution(instance, selected, details)
                    return _empty_result(
                        status="unresolved_numerical",
                        reason="existing conditional cut is violated",
                        iterations=iterations,
                        cuts=len(cuts),
                        scenarios=len(cuts),
                        statuses=tuple(solver_statuses),
                        selected=selected,
                        objective=None,
                        details={},
                        instance=instance,
                        lower_bound=lower_bound,
                        master_primal_bound=result.primal_objective,
                        master_dual_bound=result.dual_bound,
                        master_mip_gap=result.mip_gap,
                        solver_reported_optimal=True,
                        validated_feasible=validation.valid,
                        **metrics(),
                    )
                cuts.add(cut_key)
                changed = True
        objective = sum(
            (task.weight * details[task.id][0] for task in instance.tasks),
            Fraction(),
        )
        if best_objective is None or objective < best_objective:
            best_objective = objective
            best_selection = selected
            best_details = details
        if changed:
            continue
        validation_started = clock()
        validation = validate_outer_solution(instance, selected, details)
        validation_seconds = clock() - validation_started
        if not validation.valid:
            numerical_warnings.extend(validation.errors)
            return _empty_result(
                status="unresolved_validation",
                reason="; ".join(validation.errors),
                iterations=iterations,
                cuts=len(cuts),
                scenarios=len(cuts),
                statuses=tuple(solver_statuses),
                selected=selected,
                objective=None,
                details={},
                instance=instance,
                lower_bound=lower_bound,
                master_primal_bound=result.primal_objective,
                master_dual_bound=result.dual_bound,
                master_mip_gap=result.mip_gap,
                solver_reported_optimal=True,
                validated_feasible=False,
                validation_seconds=validation_seconds,
                **metrics(),
            )
        numerical_status = bool(numerical_warnings)
        # Keep the historical status value for callers, but retain the
        # warning and termination reason so this is not presented as an
        # exact-arithmetic certificate.
        status = "optimal"
        reason = (
            "no violated conditional cut"
            if not numerical_status
            else "no violation outside configured numerical tolerance"
        )
        return _empty_result(
            status=status,
            reason=reason,
            iterations=iterations,
            cuts=len(cuts),
            scenarios=len(cuts),
            statuses=tuple(solver_statuses),
            selected=selected,
            objective=objective,
            details=details,
            instance=instance,
            lower_bound=lower_bound,
            upper_bound=float(objective),
            solver_reported_optimal=True,
            validated_feasible=True,
            master_primal_bound=result.primal_objective,
            master_dual_bound=result.dual_bound,
            master_mip_gap=result.mip_gap,
            validation_seconds=validation_seconds,
            **metrics(),
        )


# Preserve the historical import surface for users of this module.
__all__ = [
    "MasterOracleOptions",
    "OuterResult",
    "TaskLocalOracleCache",
    "solve_master_oracle",
    "solve_static",
]
