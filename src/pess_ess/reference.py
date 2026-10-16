"""Deliberately independent exhaustive reference implementation."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations

from .model import Instance, Task


def _reference_matchings(
    candidate_ids: tuple[str, ...],
    endpoints: dict[str, tuple[str, str]],
) -> tuple[frozenset[str], ...]:
    result: list[frozenset[str]] = []
    for mask in range(1 << len(candidate_ids)):
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
) -> tuple[Fraction, frozenset[str]]:
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
    maximal = [
        matching
        for matching in _reference_matchings(tuple(sorted(selected_eligible)), endpoints)
        if _reference_maximal(matching, selected_eligible, endpoints)
    ]
    if not maximal:
        raise AssertionError("finite selected graph has no maximal matching")
    losses = {
        matching: abs(
            task.alpha
            + sum((task.corrections[item] for item in matching), Fraction())
        )
        for matching in maximal
    }
    value = max(losses.values())
    witness = min(
        (matching for matching, loss in losses.items() if loss == value),
        key=lambda item: tuple(sorted(item)),
    )
    return value, witness


def reference_objective(
    instance: Instance,
    selected: frozenset[str],
) -> tuple[Fraction, dict[str, tuple[Fraction, frozenset[str]]]]:
    details = {
        task.id: reference_task_loss(instance, task, selected)
        for task in instance.tasks
    }
    objective = sum(
        (task.weight * details[task.id][0] for task in instance.tasks),
        Fraction(),
    )
    return objective, details


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
