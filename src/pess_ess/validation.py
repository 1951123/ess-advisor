"""Independent feasibility and witness validation for solver outputs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction

from .matchings import is_matching, is_maximal
from .model import Instance, Task


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    maintenance_cost: Fraction
    objective: Fraction | None
    errors: tuple[str, ...]


def validate_task_witness(
    instance: Instance,
    task: Task,
    selected: Iterable[str],
    matching: Iterable[str],
    reported_loss: Fraction,
) -> tuple[str, ...]:
    selected_set = frozenset(selected)
    matching_set = frozenset(matching)
    errors: list[str] = []
    eligible = set(task.eligible_candidates)
    if not matching_set <= eligible:
        errors.append(f"{task.id}: witness contains ineligible candidate")
    if not matching_set <= selected_set:
        errors.append(f"{task.id}: witness contains unselected candidate")
    endpoints = {
        candidate_id: instance.candidate_map[candidate_id].endpoints
        for candidate_id in task.eligible_candidates
    }
    if not is_matching(matching_set, endpoints):
        errors.append(f"{task.id}: witness is not a matching")
    if not is_maximal(matching_set, selected_set & eligible, endpoints):
        errors.append(f"{task.id}: witness is not maximal")
    expected = abs(
        task.alpha + sum((task.corrections[item] for item in matching_set), Fraction())
    )
    if expected != reported_loss:
        errors.append(
            f"{task.id}: witness loss {reported_loss} != recomputed {expected}"
        )
    return tuple(errors)


def validate_outer_solution(
    instance: Instance,
    selected: Iterable[str],
    task_details: dict[str, tuple[Fraction, frozenset[str]]],
) -> ValidationResult:
    selected_tuple = tuple(selected)
    errors: list[str] = []
    if len(selected_tuple) != len(set(selected_tuple)):
        errors.append("selected resource IDs contain duplicates")
    selected_set = frozenset(selected_tuple)
    unknown = selected_set - set(instance.candidate_ids)
    if unknown:
        errors.append(f"unknown selected candidates: {sorted(unknown)}")
    maintenance = instance.maintenance_cost(selected_set) if not unknown else Fraction()
    if maintenance > instance.budget:
        errors.append(
            f"maintenance cost {maintenance} exceeds budget {instance.budget}"
        )
    expected_task_ids = {task.id for task in instance.tasks}
    if set(task_details) != expected_task_ids:
        errors.append("task details do not cover exactly the instance tasks")
    objective = Fraction()
    if not unknown:
        for task in instance.tasks:
            detail = task_details.get(task.id)
            if detail is None:
                continue
            loss, witness = detail
            errors.extend(
                validate_task_witness(
                    instance,
                    task,
                    selected_set,
                    witness,
                    loss,
                )
            )
            objective += task.weight * loss
    return ValidationResult(
        valid=not errors,
        maintenance_cost=maintenance,
        objective=objective if not errors else None,
        errors=tuple(errors),
    )
