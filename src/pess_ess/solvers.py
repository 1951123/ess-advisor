"""Static conditional and exact integer master--oracle outer solvers."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isclose
from typing import Iterable

from .matchings import enumerate_matchings
from .mip import MIPResult, solve_binary_milp
from .model import Instance, Task
from .oracle import evaluate_task, ordinary_matching_extrema
from .reference import reference_objective, solve_exhaustive


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
    occupied = set().union(
        *(set(candidates[item].endpoints) for item in matching)
    ) if matching else set()
    disjoint = frozenset(
        candidate_id
        for candidate_id in task.eligible_candidates
        if not (set(candidates[candidate_id].endpoints) & occupied)
    )
    loss = abs(
        task.alpha
        + sum((task.corrections[item] for item in matching), Fraction())
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
) -> tuple[MIPResult, dict[str, Fraction]]:
    candidate_ids = instance.candidate_ids
    task_ids = tuple(task.id for task in instance.tasks)
    x_index = {candidate_id: index for index, candidate_id in enumerate(candidate_ids)}
    z_index = {task_id: len(candidate_ids) + index for index, task_id in enumerate(task_ids)}
    lambdas = {task.id: _floor(instance, task) for task in instance.tasks}
    variable_count = len(candidate_ids) + len(task_ids)
    objective = [0.0] * len(candidate_ids) + [
        float(task.weight) for task in instance.tasks
    ]
    integrality = [1] * len(candidate_ids) + [0] * len(task_ids)
    lower_bounds = [0.0] * len(candidate_ids) + [
        float(lambdas[task.id]) for task in instance.tasks
    ]
    upper_bounds = [1.0] * len(candidate_ids) + [float("inf")] * len(task_ids)
    rows: list[tuple[dict[int, float], float, float]] = []
    rows.append(
        (
            {
                x_index[candidate.id]: float(candidate.maintenance_cost)
                for candidate in instance.candidates
            },
            -float("inf"),
            float(instance.budget),
        )
    )
    for task_id, matching_tuple in sorted(cuts):
        task = instance.task_map[task_id]
        row = _cut_row(
            instance,
            task,
            frozenset(matching_tuple),
            lambdas,
            x_index,
            z_index[task_id],
        )
        rows.append(row)
    result = solve_binary_milp(
        objective=objective,
        integrality=integrality,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        rows=rows,
        time_limit=time_limit,
    )
    return result, lambdas


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


def solve_static(
    instance: Instance,
    time_limit: float | None = None,
) -> OuterResult:
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
    result, _ = _solve_conditional_master(instance, cuts, time_limit)
    if not result.optimal:
        return OuterResult(
            status="not_optimal",
            termination_reason=result.message,
            selected=None,
            maintenance_cost=None,
            objective=None,
            lower_bound=result.objective,
            upper_bound=None,
            task_details={},
            master_iterations=1,
            distinct_cuts=len(cuts),
            inner_oracle_calls=0,
            certified_early_exits=0,
            scenario_count=sum(len(matchings) for _, matchings in all_matchings),
            solver_statuses=(result,),
        )
    selected = _extract_selection(instance, result)
    objective, details = reference_objective(instance, selected)
    return OuterResult(
        status="optimal",
        termination_reason="static full conditional formulation solved",
        selected=selected,
        maintenance_cost=instance.maintenance_cost(selected),
        objective=objective,
        lower_bound=result.objective,
        upper_bound=float(objective),
        task_details=details,
        master_iterations=1,
        distinct_cuts=len(cuts),
        inner_oracle_calls=0,
        certified_early_exits=0,
        scenario_count=sum(len(matchings) for _, matchings in all_matchings),
        solver_statuses=(result,),
    )


def solve_master_oracle(
    instance: Instance,
    time_limit: float | None = None,
    inner_time_limit: float | None = None,
    certification_tolerance: float = 0.0,
) -> OuterResult:
    cuts: set[tuple[str, tuple[str, ...]]] = set()
    solver_statuses: list[MIPResult] = []
    best_selection: frozenset[str] | None = None
    best_objective: Fraction | None = None
    best_details: dict[str, tuple[Fraction, frozenset[str]]] = {}
    lower_bound: float | None = None
    inner_calls = 0
    certified_exits = 0
    iterations = 0
    while True:
        iterations += 1
        result, _ = _solve_conditional_master(instance, cuts, time_limit)
        solver_statuses.append(result)
        lower_bound = result.objective
        if not result.optimal:
            return OuterResult(
                status="not_optimal",
                termination_reason=result.message,
                selected=best_selection,
                maintenance_cost=(
                    instance.maintenance_cost(best_selection)
                    if best_selection is not None
                    else None
                ),
                objective=best_objective,
                lower_bound=lower_bound,
                upper_bound=float(best_objective) if best_objective is not None else None,
                task_details=best_details,
                master_iterations=iterations,
                distinct_cuts=len(cuts),
                inner_oracle_calls=inner_calls,
                certified_early_exits=certified_exits,
                scenario_count=len(cuts),
                solver_statuses=tuple(solver_statuses),
            )
        selected = _extract_selection(instance, result)
        changed = False
        details: dict[str, tuple[Fraction, frozenset[str]]] = {}
        for task in instance.tasks:
            evaluation = evaluate_task(
                instance,
                task,
                selected,
                time_limit=inner_time_limit,
                certification_tolerance=certification_tolerance,
            )
            inner_calls += evaluation.mip_calls
            certified_exits += int(evaluation.certified_early_exit)
            if evaluation.source == "mip_nonoptimal":
                return OuterResult(
                    status="not_optimal",
                    termination_reason="inner MIP did not prove optimality",
                    selected=selected,
                    maintenance_cost=instance.maintenance_cost(selected),
                    objective=None,
                    lower_bound=lower_bound,
                    upper_bound=None,
                    task_details={},
                    master_iterations=iterations,
                    distinct_cuts=len(cuts),
                    inner_oracle_calls=inner_calls,
                    certified_early_exits=certified_exits,
                    scenario_count=len(cuts),
                    solver_statuses=tuple(solver_statuses)
                    + evaluation.solver_statuses,
                )
            details[task.id] = (evaluation.value, evaluation.worst_matching)
            z_value = result.values[len(instance.candidates) + list(
                task.id for task in instance.tasks
            ).index(task.id)]
            if float(evaluation.value) > z_value + 1e-7:
                cut_key = (task.id, tuple(sorted(evaluation.worst_matching)))
                if cut_key not in cuts:
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
        return OuterResult(
            status="optimal",
            termination_reason="no violated conditional cut",
            selected=selected,
            maintenance_cost=instance.maintenance_cost(selected),
            objective=objective,
            lower_bound=lower_bound,
            upper_bound=float(objective),
            task_details=details,
            master_iterations=iterations,
            distinct_cuts=len(cuts),
            inner_oracle_calls=inner_calls,
            certified_early_exits=certified_exits,
            scenario_count=len(cuts),
            solver_statuses=tuple(solver_statuses),
        )
