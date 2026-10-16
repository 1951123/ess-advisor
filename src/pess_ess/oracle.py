"""Ordinary-matching bounds and exact maximal-matching MIP oracle."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from math import lcm
from typing import Mapping

import networkx as nx

from .matchings import (
    enumerate_maximal_matchings,
    extend_to_maximal,
    is_maximal,
    is_matching,
)
from .mip import MIPResult, solve_binary_milp
from .model import Instance, Task


@dataclass(frozen=True)
class MatchingExtrema:
    minimum_value: Fraction
    minimum_matching: frozenset[str]
    maximum_value: Fraction
    maximum_matching: frozenset[str]


@dataclass(frozen=True)
class InnerBounds:
    lower_loss: Fraction
    upper_loss: Fraction
    witness_losses: tuple[tuple[frozenset[str], Fraction], ...]
    lower_witness: frozenset[str]
    minimum: MatchingExtrema


@dataclass(frozen=True)
class InnerOracleResult:
    value: Fraction
    worst_matching: frozenset[str]
    source: str
    mip_calls: int
    certified_early_exit: bool
    lower_bound: Fraction
    upper_bound: Fraction
    solver_statuses: tuple[MIPResult, ...]


def _scaled_weights(values: Mapping[str, Fraction]) -> dict[str, int]:
    denominator = lcm(*(value.denominator for value in values.values()), 1)
    return {
        key: value.numerator * (denominator // value.denominator)
        for key, value in values.items()
    }


def _networkx_extreme(
    candidates: tuple[str, ...],
    endpoints: Mapping[str, tuple[str, str]],
    weights: Mapping[str, Fraction],
    maximize: bool,
) -> frozenset[str]:
    if not candidates:
        return frozenset()
    scaled = _scaled_weights({candidate_id: weights[candidate_id] for candidate_id in candidates})
    graph = nx.Graph()
    chosen_for_pair: dict[tuple[str, str], str] = {}
    for candidate_id in sorted(candidates):
        pair = tuple(sorted(endpoints[candidate_id]))
        current = chosen_for_pair.get(pair)
        if current is None:
            chosen_for_pair[pair] = candidate_id
            continue
        if (scaled[candidate_id] > scaled[current]) if maximize else (
            scaled[candidate_id] < scaled[current]
        ):
            chosen_for_pair[pair] = candidate_id
    for pair, candidate_id in chosen_for_pair.items():
        graph.add_edge(
            pair[0],
            pair[1],
            weight=scaled[candidate_id] if maximize else -scaled[candidate_id],
            candidate_id=candidate_id,
        )
    result = nx.algorithms.matching.max_weight_matching(
        graph,
        maxcardinality=False,
        weight="weight",
    )
    return frozenset(
        graph.edges[u, v]["candidate_id"]
        for u, v in result
    )


def ordinary_matching_extrema(
    instance: Instance,
    task: Task,
    candidate_ids: tuple[str, ...] | None = None,
) -> MatchingExtrema:
    selected = tuple(
        sorted(
            task.eligible_candidates
            if candidate_ids is None
            else candidate_ids
        )
    )
    candidate_map = instance.candidate_map
    endpoints = {
        candidate_id: candidate_map[candidate_id].endpoints
        for candidate_id in selected
    }
    weights = {candidate_id: task.corrections[candidate_id] for candidate_id in selected}
    minimum_matching = _networkx_extreme(selected, endpoints, weights, maximize=False)
    maximum_matching = _networkx_extreme(selected, endpoints, weights, maximize=True)
    minimum_value = sum((weights[item] for item in minimum_matching), Fraction())
    maximum_value = sum((weights[item] for item in maximum_matching), Fraction())
    return MatchingExtrema(
        minimum_value=minimum_value,
        minimum_matching=minimum_matching,
        maximum_value=maximum_value,
        maximum_matching=maximum_matching,
    )


def _loss(task: Task, matching: frozenset[str]) -> Fraction:
    return abs(
        task.alpha
        + sum((task.corrections[item] for item in matching), Fraction())
    )


def _task_endpoints(instance: Instance, task: Task) -> dict[str, tuple[str, str]]:
    mapping = instance.candidate_map
    return {
        candidate_id: mapping[candidate_id].endpoints
        for candidate_id in task.eligible_candidates
    }


def inner_bounds(
    instance: Instance,
    task: Task,
    selected: frozenset[str],
) -> InnerBounds:
    selected_eligible = tuple(
        candidate_id
        for candidate_id in task.eligible_candidates
        if candidate_id in selected
    )
    endpoints = _task_endpoints(instance, task)
    extrema = ordinary_matching_extrema(instance, task, selected_eligible)
    upper = max(
        abs(task.alpha + extrema.minimum_value),
        abs(task.alpha + extrema.maximum_value),
    )
    witnesses = {
        extend_to_maximal(
            extrema.minimum_matching,
            selected_eligible,
            endpoints,
        ),
        extend_to_maximal(
            extrema.maximum_matching,
            selected_eligible,
            endpoints,
        ),
        extend_to_maximal(frozenset(), selected_eligible, endpoints),
    }
    witness_losses = tuple(
        sorted(
            ((matching, _loss(task, matching)) for matching in witnesses),
            key=lambda item: (item[1], tuple(sorted(item[0]))),
        )
    )
    lower_witness, lower_loss = max(
        witness_losses,
        key=lambda item: (item[1], tuple(sorted(item[0]))),
    )
    return InnerBounds(
        lower_loss=lower_loss,
        upper_loss=upper,
        witness_losses=witness_losses,
        lower_witness=lower_witness,
        minimum=extrema,
    )


def _extract_matching(
    values: tuple[float, ...],
    candidate_ids: tuple[str, ...],
    endpoints: Mapping[str, tuple[str, str]],
) -> frozenset[str]:
    matching = frozenset(
        candidate_id
        for index, candidate_id in enumerate(candidate_ids)
        if values[index] >= 1.0 - 1e-6
    )
    if not is_matching(matching, endpoints):
        raise RuntimeError("MIP returned a non-matching witness")
    return matching


def _mip_extreme(
    task: Task,
    selected: tuple[str, ...],
    endpoints: Mapping[str, tuple[str, str]],
    maximize: bool,
    time_limit: float | None,
) -> tuple[MIPResult, frozenset[str]]:
    if not selected:
        return (
            MIPResult(0, "empty selected graph", 0.0, (), None, 0),
            frozenset(),
        )
    index = {candidate_id: position for position, candidate_id in enumerate(selected)}
    objective = [
        float(-task.corrections[candidate_id] if maximize else task.corrections[candidate_id])
        for candidate_id in selected
    ]
    rows: list[tuple[dict[int, float], float, float]] = []
    vertices = sorted(
        set().union(*(set(endpoints[candidate_id]) for candidate_id in selected))
    )
    for vertex in vertices:
        incident = [
            index[candidate_id]
            for candidate_id in selected
            if vertex in endpoints[candidate_id]
        ]
        rows.append(({item: 1.0 for item in incident}, -math.inf, 1.0))
    for candidate_id in selected:
        u, v = endpoints[candidate_id]
        coefficients: dict[int, float] = {}
        for other_id in selected:
            if u in endpoints[other_id] or v in endpoints[other_id]:
                coefficients[index[other_id]] = (
                    coefficients.get(index[other_id], 0.0)
                    + (2.0 if other_id == candidate_id else 1.0)
                )
        rows.append((coefficients, 1.0, math.inf))
    result = solve_binary_milp(
        objective=objective,
        integrality=(1 for _ in selected),
        lower_bounds=(0.0 for _ in selected),
        upper_bounds=(1.0 for _ in selected),
        rows=rows,
        time_limit=time_limit,
    )
    matching = (
        _extract_matching(result.values, selected, endpoints)
        if result.values
        else frozenset()
    )
    if result.optimal and not is_maximal(matching, selected, endpoints):
        raise RuntimeError("MIP returned a non-maximal witness")
    return result, matching


def exact_maximal_matching_oracle(
    instance: Instance,
    task: Task,
    selected: frozenset[str],
    time_limit: float | None = None,
) -> InnerOracleResult:
    selected_eligible = tuple(
        candidate_id
        for candidate_id in task.eligible_candidates
        if candidate_id in selected
    )
    endpoints = _task_endpoints(instance, task)
    maximum_result, maximum_matching = _mip_extreme(
        task, selected_eligible, endpoints, True, time_limit
    )
    minimum_result, minimum_matching = _mip_extreme(
        task, selected_eligible, endpoints, False, time_limit
    )
    statuses = (maximum_result, minimum_result)
    if not all(result.optimal for result in statuses):
        return InnerOracleResult(
            value=Fraction(),
            worst_matching=frozenset(),
            source="mip_nonoptimal",
            mip_calls=2 if selected_eligible else 0,
            certified_early_exit=False,
            lower_bound=Fraction(),
            upper_bound=Fraction(),
            solver_statuses=statuses,
        )
    maximum_value = sum(
        (task.corrections[item] for item in maximum_matching),
        Fraction(),
    )
    minimum_value = sum(
        (task.corrections[item] for item in minimum_matching),
        Fraction(),
    )
    positive_side = task.alpha + maximum_value
    negative_side = -task.alpha - minimum_value
    if positive_side >= negative_side:
        value = positive_side
        worst_matching = maximum_matching
    else:
        value = negative_side
        worst_matching = minimum_matching
    bounds = inner_bounds(instance, task, selected)
    return InnerOracleResult(
        value=value,
        worst_matching=worst_matching,
        source="mip",
        mip_calls=2 if selected_eligible else 0,
        certified_early_exit=False,
        lower_bound=bounds.lower_loss,
        upper_bound=bounds.upper_loss,
        solver_statuses=statuses,
    )


def evaluate_task(
    instance: Instance,
    task: Task,
    selected: frozenset[str],
    time_limit: float | None = None,
    certification_tolerance: float = 0.0,
) -> InnerOracleResult:
    bounds = inner_bounds(instance, task, selected)
    exact_equal = bounds.lower_loss == bounds.upper_loss
    close = math.isclose(
        float(bounds.lower_loss),
        float(bounds.upper_loss),
        rel_tol=certification_tolerance,
        abs_tol=certification_tolerance,
    )
    if exact_equal or (certification_tolerance > 0 and close):
        return InnerOracleResult(
            value=bounds.upper_loss,
            worst_matching=bounds.lower_witness,
            source="certified_bounds",
            mip_calls=0,
            certified_early_exit=True,
            lower_bound=bounds.lower_loss,
            upper_bound=bounds.upper_loss,
            solver_statuses=(),
        )
    return exact_maximal_matching_oracle(
        instance,
        task,
        selected,
        time_limit=time_limit,
    )
