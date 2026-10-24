"""Exact-objective, strict-improvement add-only greedy resource selection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from time import monotonic

from .model import Instance
from .reference import (
    ReferenceEvaluationIncomplete,
    ReferenceWork,
    reference_nominal_objective,
    reference_objective,
)


@dataclass(frozen=True)
class GreedyStep:
    candidate_id: str
    objective_before: Fraction
    objective_after: Fraction
    improvement: Fraction


@dataclass(frozen=True)
class AddOnlyGreedyResult:
    objective: Fraction | None
    selected: frozenset[str]
    budget: Fraction
    maintenance_cost: Fraction
    unused_budget: Fraction
    steps: tuple[GreedyStep, ...]
    status: str
    termination_reason: str
    termination_detail: str | None
    candidate_attempts: int
    fixed_set_evaluation_calls: int
    completed_fixed_set_evaluations: int
    inner_matching_subsets_examined: int
    inner_maximal_matchings_scored: int
    elapsed_seconds: float
    priority_order: tuple[str, ...] | None

    @property
    def addition_order(self) -> tuple[str, ...]:
        return tuple(step.candidate_id for step in self.steps)


def _pess_matching_subset_work(instance: Instance, selected: frozenset[str]) -> int:
    return sum(
        1 << len(set(task.eligible_candidates) & selected)
        for task in instance.tasks
    )


def _check_nonnegative_integer(name: str, value: int | None) -> None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, int) or value < 0
    ):
        raise ValueError(f"{name} must be a nonnegative integer")


def solve_add_only_greedy(
    instance: Instance,
    *,
    objective: str,
    priority_order: tuple[str, ...] | list[str] | None = None,
    total_time_limit: float | None = None,
    max_evaluations: int | None = None,
    max_inner_matching_subsets: int | None = None,
) -> AddOnlyGreedyResult:
    """Run strict-improvement best-addition greedy with exact fixed-set values.

    ``max_evaluations`` is a shared cap on exact fixed-set objective calls,
    including the initial empty-set evaluation. For PESS,
    ``max_inner_matching_subsets`` additionally caps the subset checks in the
    exact maximal-matching reference evaluator. A round is committed only
    after every remaining budget-feasible candidate has an exact evaluation.
    """
    if objective not in {"pess", "nominal"}:
        raise ValueError("objective must be 'pess' or 'nominal'")
    if total_time_limit is not None and (
        isinstance(total_time_limit, bool)
        or not isinstance(total_time_limit, (int, float))
        or not math.isfinite(total_time_limit)
        or total_time_limit < 0
    ):
        raise ValueError("total_time_limit must be a finite nonnegative number")
    _check_nonnegative_integer("max_evaluations", max_evaluations)
    _check_nonnegative_integer(
        "max_inner_matching_subsets", max_inner_matching_subsets
    )

    if objective == "nominal":
        if priority_order is None:
            raise ValueError("objective='nominal' requires priority_order")
        from .nominal import validate_priority_order

        resolved_priority = validate_priority_order(instance, priority_order)
        if max_inner_matching_subsets is not None:
            raise ValueError(
                "max_inner_matching_subsets applies only to objective='pess'"
            )
    else:
        if priority_order is not None:
            raise ValueError("priority_order is supported only by objective='nominal'")
        resolved_priority = None

    started = monotonic()
    deadline = (
        started + total_time_limit if total_time_limit is not None else None
    )
    selected = frozenset()
    current_objective: Fraction | None = None
    current_cost = Fraction()
    steps: list[GreedyStep] = []
    candidate_attempts = 0
    fixed_set_evaluation_calls = 0
    completed_fixed_set_evaluations = 0
    inner_matching_subsets_examined = 0
    inner_maximal_matchings_scored = 0
    termination_reason = "no_strict_improvement"
    termination_detail: str | None = None

    def result() -> AddOnlyGreedyResult:
        elapsed = max(0.0, monotonic() - started)
        status = (
            "heuristic_incomplete"
            if termination_reason == "evaluation_incomplete"
            else "heuristic_completed"
        )
        return AddOnlyGreedyResult(
            objective=current_objective,
            selected=selected,
            budget=instance.budget,
            maintenance_cost=current_cost,
            unused_budget=instance.budget - current_cost,
            steps=tuple(steps),
            status=status,
            termination_reason=termination_reason,
            termination_detail=termination_detail,
            candidate_attempts=candidate_attempts,
            fixed_set_evaluation_calls=fixed_set_evaluation_calls,
            completed_fixed_set_evaluations=completed_fixed_set_evaluations,
            inner_matching_subsets_examined=inner_matching_subsets_examined,
            inner_maximal_matchings_scored=inner_maximal_matchings_scored,
            elapsed_seconds=elapsed,
            priority_order=resolved_priority,
        )

    def incomplete(detail: str) -> None:
        nonlocal termination_reason, termination_detail
        termination_reason = "evaluation_incomplete"
        termination_detail = detail

    def evaluate(
        selection: frozenset[str],
    ) -> tuple[Fraction | None, str | None]:
        nonlocal fixed_set_evaluation_calls
        nonlocal completed_fixed_set_evaluations
        nonlocal inner_matching_subsets_examined
        nonlocal inner_maximal_matchings_scored

        if max_evaluations is not None and fixed_set_evaluation_calls >= max_evaluations:
            return None, "max_evaluations"
        if deadline is not None and monotonic() >= deadline:
            return None, "total_time_limit"

        work_required = (
            _pess_matching_subset_work(instance, selection)
            if objective == "pess"
            else 0
        )
        if (
            max_inner_matching_subsets is not None
            and inner_matching_subsets_examined + work_required
            > max_inner_matching_subsets
        ):
            return None, "max_inner_matching_subsets"

        fixed_set_evaluation_calls += 1
        work = ReferenceWork()
        try:
            if objective == "pess":
                value, _ = reference_objective(
                    instance,
                    selection,
                    deadline=deadline,
                    work=work,
                )
            else:
                value, _ = reference_nominal_objective(
                    instance,
                    selection,
                    resolved_priority or (),
                    deadline=deadline,
                )
        except ReferenceEvaluationIncomplete:
            inner_matching_subsets_examined += work.matching_subsets_examined
            inner_maximal_matchings_scored += work.maximal_matchings_scored
            return None, "total_time_limit"
        except TimeoutError:
            inner_matching_subsets_examined += work.matching_subsets_examined
            inner_maximal_matchings_scored += work.maximal_matchings_scored
            return None, "inner_evaluation_incomplete"

        inner_matching_subsets_examined += work.matching_subsets_examined
        inner_maximal_matchings_scored += work.maximal_matchings_scored
        completed_fixed_set_evaluations += 1
        if deadline is not None and monotonic() >= deadline:
            return None, "total_time_limit"
        return value, None

    initial_objective, detail = evaluate(selected)
    if detail is not None:
        incomplete(detail)
        return result()
    current_objective = initial_objective
    assert current_objective is not None

    candidate_ids = tuple(sorted(instance.candidate_ids))
    costs = {candidate.id: candidate.maintenance_cost for candidate in instance.candidates}
    while True:
        feasible_candidates: list[str] = []
        for candidate_id in candidate_ids:
            if candidate_id in selected:
                continue
            if deadline is not None and monotonic() >= deadline:
                incomplete("total_time_limit")
                return result()
            candidate_attempts += 1
            if current_cost + costs[candidate_id] <= instance.budget:
                feasible_candidates.append(candidate_id)

        if not feasible_candidates:
            termination_reason = "no_feasible_candidate"
            return result()

        best_candidate: str | None = None
        best_objective: Fraction | None = None
        best_improvement = Fraction()
        for candidate_id in feasible_candidates:
            candidate_value, detail = evaluate(selected | {candidate_id})
            if detail is not None:
                incomplete(detail)
                return result()
            assert candidate_value is not None
            improvement = current_objective - candidate_value
            if improvement > best_improvement:
                best_candidate = candidate_id
                best_objective = candidate_value
                best_improvement = improvement

        if best_candidate is None or best_objective is None:
            termination_reason = "no_strict_improvement"
            return result()

        steps.append(
            GreedyStep(
                candidate_id=best_candidate,
                objective_before=current_objective,
                objective_after=best_objective,
                improvement=best_improvement,
            )
        )
        selected = selected | {best_candidate}
        current_cost += costs[best_candidate]
        current_objective = best_objective
