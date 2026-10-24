"""Deliberately independent exhaustive reference implementation."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from time import monotonic

from .model import Instance, Task


class ReferenceEvaluationIncomplete(TimeoutError):
    """An exact reference evaluation stopped before producing a value."""


@dataclass
class ReferenceWork:
    """Optional work counters for exact PESS reference evaluations."""

    matching_subsets_examined: int = 0
    maximal_matchings_scored: int = 0


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and monotonic() >= deadline:
        raise ReferenceEvaluationIncomplete("total_time_limit")


def _reference_matchings(
    candidate_ids: tuple[str, ...],
    endpoints: dict[str, tuple[str, str]],
    *,
    deadline: float | None = None,
    work: ReferenceWork | None = None,
) -> tuple[frozenset[str], ...]:
    result: list[frozenset[str]] = []
    for mask in range(1 << len(candidate_ids)):
        _check_deadline(deadline)
        if work is not None:
            work.matching_subsets_examined += 1
        chosen = frozenset(
            candidate_ids[index]
            for index in range(len(candidate_ids))
            if mask & (1 << index)
        )
        occupied: set[str] = set()
        valid = True
        for candidate_id in chosen:
            edge = set(endpoints[candidate_id])
            if occupied & edge:
                valid = False
                break
            occupied.update(edge)
        if valid:
            result.append(chosen)
    return tuple(sorted(result, key=lambda item: tuple(sorted(item))))


def _reference_maximal(
    matching: frozenset[str],
    selected: frozenset[str],
    endpoints: dict[str, tuple[str, str]],
) -> bool:
    occupied = set()
    for candidate_id in matching:
        edge = set(endpoints[candidate_id])
        if occupied & edge:
            return False
        occupied.update(edge)
    if not matching <= selected:
        return False
    return all(
        occupied & set(endpoints[candidate_id])
        for candidate_id in selected - matching
    )


def reference_task_loss(
    instance: Instance,
    task: Task,
    selected: frozenset[str],
    *,
    deadline: float | None = None,
    work: ReferenceWork | None = None,
) -> tuple[Fraction, frozenset[str]]:
    _check_deadline(deadline)
    candidate_map = instance.candidate_map
    selected_eligible = frozenset(
        candidate_id
        for candidate_id in task.eligible_candidates
        if candidate_id in selected
    )
    endpoints = {
        candidate_id: candidate_map[candidate_id].endpoints
        for candidate_id in task.eligible_candidates
    }
    matchings = _reference_matchings(
        tuple(sorted(selected_eligible)),
        endpoints,
        deadline=deadline,
        work=work,
    )
    maximal = []
    for matching in matchings:
        _check_deadline(deadline)
        if _reference_maximal(matching, selected_eligible, endpoints):
            maximal.append(matching)
    if not maximal:
        raise AssertionError("finite selected graph has no maximal matching")
    losses = {}
    for matching in maximal:
        _check_deadline(deadline)
        losses[matching] = abs(
            task.alpha
            + sum((task.corrections[item] for item in matching), Fraction())
        )
        if work is not None:
            work.maximal_matchings_scored += 1
    value = max(losses.values())
    witness = min(
        (matching for matching, loss in losses.items() if loss == value),
        key=lambda item: tuple(sorted(item)),
    )
    return value, witness


def reference_objective(
    instance: Instance,
    selected: frozenset[str],
    *,
    deadline: float | None = None,
    work: ReferenceWork | None = None,
) -> tuple[Fraction, dict[str, tuple[Fraction, frozenset[str]]]]:
    details = {}
    objective = Fraction()
    for task in instance.tasks:
        _check_deadline(deadline)
        details[task.id] = reference_task_loss(
            instance, task, selected, deadline=deadline, work=work
        )
        objective += task.weight * details[task.id][0]
    _check_deadline(deadline)
    return objective, details


def reference_nominal_task_loss(
    instance: Instance,
    task: Task,
    selected: frozenset[str],
    priority_order: tuple[str, ...],
    *,
    deadline: float | None = None,
) -> tuple[Fraction, frozenset[str]]:
    """Independent exact replay of a fixed global-priority greedy policy."""
    candidate_map = instance.candidate_map
    eligible = set(task.eligible_candidates)
    occupied: set[str] = set()
    matching: set[str] = set()
    for candidate_id in priority_order:
        _check_deadline(deadline)
        if candidate_id not in selected or candidate_id not in eligible:
            continue
        endpoints = candidate_map[candidate_id].endpoints
        if endpoints[0] not in occupied and endpoints[1] not in occupied:
            matching.add(candidate_id)
            occupied.add(endpoints[0])
            occupied.add(endpoints[1])
    matching_set = frozenset(matching)
    loss = abs(
        task.alpha
        + sum((task.corrections[item] for item in matching_set), Fraction())
    )
    return loss, matching_set


def reference_nominal_objective(
    instance: Instance,
    selected: frozenset[str],
    priority_order: tuple[str, ...],
    *,
    deadline: float | None = None,
) -> tuple[Fraction, dict[str, tuple[Fraction, frozenset[str]]]]:
    details = {}
    objective = Fraction()
    for task in instance.tasks:
        _check_deadline(deadline)
        details[task.id] = reference_nominal_task_loss(
            instance, task, selected, priority_order, deadline=deadline
        )
        objective += task.weight * details[task.id][0]
    _check_deadline(deadline)
    return objective, details


@dataclass(frozen=True)
class NominalReferenceResult:
    objective: Fraction
    optimal_sets: tuple[frozenset[str], ...]
    evaluated_subsets: int
    feasible_subsets: int


def solve_nominal_exhaustive(
    instance: Instance,
    priority_order: tuple[str, ...],
) -> NominalReferenceResult:
    """Tiny-instance exact reference; production solving uses a MILP."""
    candidate_ids = instance.candidate_ids
    best: Fraction | None = None
    optimal: list[frozenset[str]] = []
    feasible = 0
    evaluated = 0
    for size in range(len(candidate_ids) + 1):
        for selected_tuple in combinations(candidate_ids, size):
            evaluated += 1
            selected = frozenset(selected_tuple)
            if instance.maintenance_cost(selected) > instance.budget:
                continue
            feasible += 1
            objective, _ = reference_nominal_objective(
                instance, selected, priority_order
            )
            if best is None or objective < best:
                best = objective
                optimal = [selected]
            elif objective == best:
                optimal.append(selected)
    if best is None:
        raise AssertionError("empty selection should be budget feasible")
    return NominalReferenceResult(
        objective=best,
        optimal_sets=tuple(sorted(optimal, key=lambda item: tuple(sorted(item)))),
        evaluated_subsets=evaluated,
        feasible_subsets=feasible,
    )


@dataclass(frozen=True)
class ReferenceResult:
    objective: Fraction
    optimal_sets: tuple[frozenset[str], ...]
    evaluated_subsets: int
    feasible_subsets: int


def solve_exhaustive(instance: Instance) -> ReferenceResult:
    candidate_ids = instance.candidate_ids
    best: Fraction | None = None
    optimal: list[frozenset[str]] = []
    feasible = 0
    evaluated = 0
    for size in range(len(candidate_ids) + 1):
        for selected_tuple in combinations(candidate_ids, size):
            evaluated += 1
            selected = frozenset(selected_tuple)
            if instance.maintenance_cost(selected) > instance.budget:
                continue
            feasible += 1
            objective, _ = reference_objective(instance, selected)
            if best is None or objective < best:
                best = objective
                optimal = [selected]
            elif objective == best:
                optimal.append(selected)
    if best is None:
        raise AssertionError("empty selection should be budget feasible")
    return ReferenceResult(
        objective=best,
        optimal_sets=tuple(sorted(optimal, key=lambda item: tuple(sorted(item)))),
        evaluated_subsets=evaluated,
        feasible_subsets=feasible,
    )
