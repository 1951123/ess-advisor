"""Matching enumeration and independent structural predicates."""

from __future__ import annotations

from collections.abc import Iterable, Mapping


def _endpoint_set(candidate_id: str, endpoints: Mapping[str, tuple[str, str]]) -> set[str]:
    return set(endpoints[candidate_id])


def is_matching(
    matching: Iterable[str],
    endpoints: Mapping[str, tuple[str, str]],
) -> bool:
    used: set[str] = set()
    for candidate_id in matching:
        edge_endpoints = _endpoint_set(candidate_id, endpoints)
        if used & edge_endpoints:
            return False
        used |= edge_endpoints
    return True


def is_maximal(
    matching: Iterable[str],
    selected: Iterable[str],
    endpoints: Mapping[str, tuple[str, str]],
) -> bool:
    matching_set = frozenset(matching)
    selected_set = frozenset(selected)
    if not matching_set <= selected_set or not is_matching(matching_set, endpoints):
        return False
    occupied = set().union(
        *(_endpoint_set(candidate_id, endpoints) for candidate_id in matching_set)
    ) if matching_set else set()
    for candidate_id in sorted(selected_set - matching_set):
        if not (occupied & _endpoint_set(candidate_id, endpoints)):
            return False
    return True


def enumerate_matchings(
    candidate_ids: Iterable[str],
    endpoints: Mapping[str, tuple[str, str]],
) -> tuple[frozenset[str], ...]:
    """Enumerate every matching using an independent include/skip recursion."""
    ordered = tuple(sorted(candidate_ids))
    result: list[frozenset[str]] = []

    def visit(position: int, chosen: tuple[str, ...], occupied: frozenset[str]) -> None:
        if position == len(ordered):
            result.append(frozenset(chosen))
            return
        candidate_id = ordered[position]
        visit(position + 1, chosen, occupied)
        edge_endpoints = frozenset(endpoints[candidate_id])
        if not occupied & edge_endpoints:
            visit(
                position + 1,
                chosen + (candidate_id,),
                occupied | edge_endpoints,
            )

    visit(0, (), frozenset())
    return tuple(sorted(result, key=lambda item: tuple(sorted(item))))


def enumerate_maximal_matchings(
    selected: Iterable[str],
    endpoints: Mapping[str, tuple[str, str]],
) -> tuple[frozenset[str], ...]:
    selected_set = frozenset(selected)
    return tuple(
        matching
        for matching in enumerate_matchings(selected_set, endpoints)
        if is_maximal(matching, selected_set, endpoints)
    )


def extend_to_maximal(
    matching: Iterable[str],
    selected: Iterable[str],
    endpoints: Mapping[str, tuple[str, str]],
) -> frozenset[str]:
    """Greedily extend a matching; this returns a witness, not an optimum."""
    current = set(matching)
    selected_set = set(selected)
    if not is_matching(current, endpoints) or not current <= selected_set:
        raise ValueError("matching must be a matching contained in selected edges")
    while True:
        occupied = set().union(
            *(set(endpoints[candidate_id]) for candidate_id in current)
        ) if current else set()
        additions = [
            candidate_id
            for candidate_id in sorted(selected_set - current)
            if not (occupied & set(endpoints[candidate_id]))
        ]
        if not additions:
            return frozenset(current)
        current.add(additions[0])
