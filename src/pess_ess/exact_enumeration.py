"""Budget-pruned exact outer resource-subset enumeration."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import lcm
from time import monotonic

from .model import Instance


@dataclass(frozen=True)
class ExactEnumerationResult:
    objective: Fraction | None
    optimal_sets: tuple[frozenset[str], ...]
    evaluated_subsets: int
    feasible_subsets: int
    pruned_branches: int
    complete: bool
    elapsed_seconds: float


@dataclass(frozen=True)
class _CompiledTask:
    alpha: int
    corrections: tuple[int, ...]
    eligible_mask: int
    weight: int


def _compile(instance: Instance) -> tuple[int, int, tuple[int, ...], tuple[_CompiledTask, ...]]:
    """Compile rational data to common integer scales without rounding."""
    candidates = instance.candidates
    candidate_index = {item.id: i for i, item in enumerate(candidates)}
    coefficient_scale = 1
    weight_scale = 1
    for task in instance.tasks:
        coefficient_scale = lcm(coefficient_scale, task.alpha.denominator)
        weight_scale = lcm(weight_scale, task.weight.denominator)
        for value in task.corrections.values():
            coefficient_scale = lcm(coefficient_scale, value.denominator)
    conflicts = []
    for candidate in candidates:
        mask = 0
        for other_index, other in enumerate(candidates):
            if set(candidate.endpoints) & set(other.endpoints):
                mask |= 1 << other_index
        conflicts.append(mask)
    compiled = []
    for task in instance.tasks:
        eligible_mask = sum(1 << candidate_index[item] for item in task.eligible_candidates)
        corrections = tuple(
            int(task.corrections.get(candidate.id, Fraction()) * coefficient_scale)
            for candidate in candidates
        )
        compiled.append(_CompiledTask(
            alpha=int(task.alpha * coefficient_scale),
            corrections=corrections,
            eligible_mask=eligible_mask,
            weight=int(task.weight * weight_scale),
        ))
    return coefficient_scale, weight_scale, tuple(conflicts), tuple(compiled)


def _compiled_objective(
    chosen_mask: int,
    tasks: tuple[_CompiledTask, ...],
    conflicts: tuple[int, ...],
    priority_indices: tuple[int, ...] | None,
    deadline: float | None,
) -> int | None:
    total = 0
    for task in tasks:
        if deadline is not None and monotonic() >= deadline:
            return None
        active = chosen_mask & task.eligible_mask
        if priority_indices is not None:
            matching = 0
            for index in priority_indices:
                bit = 1 << index
                if active & bit and not matching & conflicts[index]:
                    matching |= bit
            correction = 0
            bits = matching
            while bits:
                bit = bits & -bits
                correction += task.corrections[bit.bit_length() - 1]
                bits ^= bit
            loss = abs(task.alpha + correction)
        else:
            active_count = active.bit_count()
            if active_count == 0:
                loss = abs(task.alpha)
            elif active_count == 1:
                index = active.bit_length() - 1
                loss = abs(task.alpha + task.corrections[index])
            elif active_count == 2:
                first = active & -active
                second = active ^ first
                i, j = first.bit_length() - 1, second.bit_length() - 1
                if conflicts[i] & second:
                    loss = max(
                        abs(task.alpha + task.corrections[i]),
                        abs(task.alpha + task.corrections[j]),
                    )
                else:
                    loss = abs(
                        task.alpha + task.corrections[i] + task.corrections[j]
                    )
            else:
                best_loss = -1
                matching = active
                while True:
                    bits = matching
                    valid = True
                    while bits:
                        bit = bits & -bits
                        index = bit.bit_length() - 1
                        if bits != bit and (matching ^ bit) & conflicts[index]:
                            valid = False
                            break
                        bits ^= bit
                    if valid:
                        unmatched = active & ~matching
                        maximal = True
                        while unmatched:
                            bit = unmatched & -unmatched
                            index = bit.bit_length() - 1
                            if not matching & conflicts[index]:
                                maximal = False
                                break
                            unmatched ^= bit
                        if maximal:
                            bits = matching
                            correction = 0
                            while bits:
                                bit = bits & -bits
                                index = bit.bit_length() - 1
                                correction += task.corrections[index]
                                bits ^= bit
                            best_loss = max(best_loss, abs(task.alpha + correction))
                    if matching == 0:
                        break
                    matching = (matching - 1) & active
                loss = best_loss
        total += task.weight * loss
    return total


def solve_exact_enumeration(
    instance: Instance,
    *,
    objective: str,
    priority_order: tuple[str, ...] | None = None,
    time_limit: float | None = None,
) -> ExactEnumerationResult:
    """Enumerate exactly the budget-feasible selections, pruning by cost.

    Nonnegative costs make a branch infeasible as soon as its partial sum
    exceeds the budget. A timeout is checked between objective evaluations;
    its incumbent/ties are returned without an exact-optimality claim.
    """
    if objective not in {"pess", "nominal"}:
        raise ValueError("objective must be 'pess' or 'nominal'")
    if objective == "nominal":
        if priority_order is None:
            raise ValueError("objective='nominal' requires priority_order")
        if (
            len(priority_order) != len(instance.candidate_ids)
            or set(priority_order) != set(instance.candidate_ids)
        ):
            raise ValueError("priority_order must be a permutation of all candidate ids")
    elif priority_order is not None:
        raise ValueError("priority_order is supported only by objective='nominal'")
    if time_limit is not None and time_limit < 0:
        raise ValueError("time_limit must be nonnegative")

    started = monotonic()
    deadline = started + time_limit if time_limit is not None else None
    ids = instance.candidate_ids
    costs = tuple(item.maintenance_cost for item in instance.candidates)
    coefficient_scale, weight_scale, conflicts, compiled_tasks = _compile(instance)
    priority_indices = (
        tuple(ids.index(item) for item in priority_order)
        if objective == "nominal" and priority_order is not None
        else None
    )
    selected: list[str] = []
    best: Fraction | None = None
    ties: list[frozenset[str]] = []
    evaluated = feasible = pruned = 0
    complete = True

    def visit(index: int, cost: Fraction) -> None:
        nonlocal best, ties, evaluated, feasible, pruned, complete
        if deadline is not None and monotonic() >= deadline:
            complete = False
            return
        if cost > instance.budget:
            pruned += 1
            return
        if index == len(ids):
            chosen = frozenset(selected)
            feasible += 1
            chosen_mask = sum(1 << index for index, item in enumerate(ids) if item in chosen)
            selected_priority = (
                tuple(
                    index
                    for index in priority_indices or ()
                    if chosen_mask & (1 << index)
                )
                if priority_indices is not None
                else None
            )
            value_integer = _compiled_objective(
                chosen_mask, compiled_tasks, conflicts, selected_priority, deadline
            )
            if value_integer is None:
                complete = False
                return
            evaluated += 1
            value = Fraction(value_integer, coefficient_scale * weight_scale)
            if best is None or value < best:
                best, ties = value, [chosen]
            elif value == best:
                ties.append(chosen)
            return

        visit(index + 1, cost)
        if not complete:
            return
        selected.append(ids[index])
        visit(index + 1, cost + costs[index])
        selected.pop()

    visit(0, Fraction())
    elapsed = monotonic() - started
    ordered = tuple(sorted(ties, key=lambda item: tuple(sorted(item))))
    return ExactEnumerationResult(
        objective=best,
        optimal_sets=ordered,
        evaluated_subsets=evaluated,
        feasible_subsets=feasible,
        pruned_branches=pruned,
        complete=complete,
        elapsed_seconds=elapsed,
    )
