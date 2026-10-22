"""Fixed-priority nominal optimization for labeled matching resources."""

from __future__ import annotations

import time
from collections.abc import Iterable
from fractions import Fraction
from math import isfinite

from .mip import MIPResult, solve_binary_milp
from .model import Instance, Task
from .solvers import OuterResult
from .validation import validate_outer_solution


def validate_priority_order(
    instance: Instance, priority_order: Iterable[str]
) -> tuple[str, ...]:
    """Require an explicit total order over every candidate ID."""
    if isinstance(priority_order, (str, bytes)):
        raise TypeError("priority_order must be a sequence of candidate IDs")
    order = tuple(priority_order)
    if any(not isinstance(candidate_id, str) for candidate_id in order):
        raise TypeError("priority_order entries must be strings")
    if len(order) != len(set(order)):
        raise ValueError("priority_order must not contain duplicate candidate IDs")
    if set(order) != set(instance.candidate_ids):
        missing = sorted(set(instance.candidate_ids) - set(order))
        unknown = sorted(set(order) - set(instance.candidate_ids))
        raise ValueError(
            f"priority_order must be a permutation of candidate IDs; "
            f"missing={missing}, unknown={unknown}"
        )
    return order


def priority_matching(
    instance: Instance,
    task: Task,
    selected: frozenset[str],
    priority_order: tuple[str, ...],
) -> frozenset[str]:
    """Replay the shared fixed-priority greedy maximal-matching policy."""
    candidates = instance.candidate_map
    eligible = set(task.eligible_candidates)
    matching: set[str] = set()
    occupied: set[str] = set()
    for candidate_id in priority_order:
        if candidate_id not in selected or candidate_id not in eligible:
            continue
        endpoints = set(candidates[candidate_id].endpoints)
        if not occupied.intersection(endpoints):
            matching.add(candidate_id)
            occupied.update(endpoints)
    return frozenset(matching)


def _task_details(
    instance: Instance,
    selected: frozenset[str],
    priority_order: tuple[str, ...],
) -> dict[str, tuple[Fraction, frozenset[str]]]:
    details = {}
    for task in instance.tasks:
        matching = priority_matching(instance, task, selected, priority_order)
        loss = abs(
            task.alpha
            + sum((task.corrections[candidate_id] for candidate_id in matching), Fraction())
        )
        details[task.id] = (loss, matching)
    return details


def _extract_selection(result: MIPResult, candidate_ids: tuple[str, ...]) -> frozenset[str]:
    if len(result.values) < len(candidate_ids):
        raise RuntimeError("MIP did not return a complete selection vector")
    selected = set()
    for index, candidate_id in enumerate(candidate_ids):
        value = result.values[index]
        if not isfinite(value):
            raise RuntimeError("MIP returned a non-finite resource selection")
        if abs(value - round(value)) > 1e-5:
            raise RuntimeError("MIP returned a fractional resource selection")
        if round(value):
            selected.add(candidate_id)
    return frozenset(selected)


def _lower_bound(result: MIPResult) -> float | None:
    if result.dual_bound is not None:
        return result.dual_bound
    return result.primal_objective if result.optimal else None


def _nominal_milp(
    instance: Instance,
    priority_order: tuple[str, ...],
    time_limit: float | None,
    deadline: float | None,
    clock,
) -> tuple[MIPResult, float]:
    build_started = clock()
    candidate_ids = instance.candidate_ids
    x_index = {candidate_id: index for index, candidate_id in enumerate(candidate_ids)}

    priority_rank = {candidate_id: index for index, candidate_id in enumerate(priority_order)}
    task_order = {
        task.id: tuple(sorted(task.eligible_candidates, key=priority_rank.__getitem__))
        for task in instance.tasks
    }
    y_index: dict[tuple[str, str], int] = {}
    next_index = len(candidate_ids)
    for task in instance.tasks:
        for candidate_id in task_order[task.id]:
            y_index[(task.id, candidate_id)] = next_index
            next_index += 1
    z_index = {task.id: next_index + index for index, task in enumerate(instance.tasks)}
    variable_count = next_index + len(instance.tasks)

    objective = [0.0] * variable_count
    integrality = [0] * variable_count
    lower_bounds = [0.0] * variable_count
    upper_bounds = [float("inf")] * variable_count
    for candidate_id, index in x_index.items():
        integrality[index] = 1
        upper_bounds[index] = 1.0
    for index in y_index.values():
        integrality[index] = 1
        upper_bounds[index] = 1.0
    for task in instance.tasks:
        objective[z_index[task.id]] = float(task.weight)

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
    candidates = instance.candidate_map
    for task in instance.tasks:
        ordered = task_order[task.id]
        for position, candidate_id in enumerate(ordered):
            current = y_index[(task.id, candidate_id)]
            # Select an edge in the matching only if its resource is selected.
            rows.append(({current: 1.0, x_index[candidate_id]: -1.0}, -float("inf"), 0.0))

            earlier_conflicts = [
                earlier_id
                for earlier_id in ordered[:position]
                if set(candidates[earlier_id].endpoints)
                & set(candidates[candidate_id].endpoints)
            ]
            # A selected earlier edge blocks the current edge.
            for earlier_id in earlier_conflicts:
                earlier = y_index[(task.id, earlier_id)]
                rows.append(({current: 1.0, earlier: 1.0}, -float("inf"), 1.0))
            # If selected, the current edge must be used unless an earlier
            # selected edge already blocks it. Together these rows encode the
            # deterministic greedy scan for binary resource and matching vars.
            lower_row = {current: 1.0, x_index[candidate_id]: -1.0}
            for earlier_id in earlier_conflicts:
                earlier = y_index[(task.id, earlier_id)]
                lower_row[earlier] = lower_row.get(earlier, 0.0) + 1.0
            rows.append((lower_row, 0.0, float("inf")))

        z = z_index[task.id]
        positive_row = {z: 1.0}
        negative_row = {z: 1.0}
        for candidate_id in ordered:
            coefficient = float(task.corrections[candidate_id])
            y = y_index[(task.id, candidate_id)]
            positive_row[y] = -coefficient
            negative_row[y] = coefficient
        rows.append((positive_row, float(task.alpha), float("inf")))
        rows.append((negative_row, float(-task.alpha), float("inf")))

    build_seconds = clock() - build_started
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
    return result, build_seconds


def solve_nominal(
    instance: Instance,
    priority_order: Iterable[str],
    *,
    time_limit: float | None = None,
    total_time_limit: float | None = None,
    clock=time.perf_counter,
) -> OuterResult:
    """Minimize exact model loss under a known global greedy priority order."""
    order = validate_priority_order(instance, priority_order)
    for name, limit in (
        ("time_limit", time_limit),
        ("total_time_limit", total_time_limit),
    ):
        if limit is not None and (
            isinstance(limit, bool)
            or not isinstance(limit, (int, float))
            or not isfinite(limit)
            or limit < 0
        ):
            raise ValueError(f"{name} must be a finite nonnegative number")
    started = clock()
    deadline = started + total_time_limit if total_time_limit is not None else None
    mip, build_seconds = _nominal_milp(
        instance, order, time_limit, deadline, clock
    )
    selected: frozenset[str] | None = None
    details: dict[str, tuple[Fraction, frozenset[str]]] = {}
    validation = None
    if mip.incumbent_returned:
        try:
            selected = _extract_selection(mip, instance.candidate_ids)
            if instance.maintenance_cost(selected) <= instance.budget:
                details = _task_details(instance, selected, order)
                validation = validate_outer_solution(instance, selected, details)
                if not validation.valid:
                    selected = None
                    details = {}
            else:
                selected = None
        except (RuntimeError, ValueError, OverflowError):
            selected = None
            details = {}

    valid = bool(validation and validation.valid and selected is not None)
    if not mip.optimal:
        status = "feasible_unproven" if valid else "not_optimal"
        reason = (
            "nonoptimal fixed-priority incumbent replayed and validated; "
            "global optimality not established"
            if valid
            else mip.message
        )
    else:
        status = "optimal" if valid else "unresolved_validation"
        reason = (
            "fixed-priority MILP reported optimality; selected policy matching "
            "replayed and validated"
            if valid
            else ("; ".join(validation.errors) if validation else mip.message)
        )
    objective = validation.objective if valid and validation else None
    return OuterResult(
        status=status,
        termination_reason=reason,
        selected=selected,
        maintenance_cost=(
            instance.maintenance_cost(selected) if selected is not None else None
        ),
        objective=objective,
        lower_bound=_lower_bound(mip),
        upper_bound=float(objective) if objective is not None else None,
        task_details=details,
        master_iterations=1,
        distinct_cuts=0,
        inner_oracle_calls=0,
        certified_early_exits=0,
        scenario_count=0,
        solver_statuses=(mip,),
        solver_reported_optimal=mip.optimal,
        validated_feasible=valid,
        master_primal_bound=mip.primal_objective,
        master_dual_bound=mip.dual_bound,
        master_mip_gap=mip.mip_gap,
        master_phase_seconds=mip.total_duration_seconds,
        cut_construction_seconds=build_seconds,
        total_runtime_seconds=clock() - started,
        highs_invocations=mip.highs_invocations,
        policy_priority_order=order,
    )
