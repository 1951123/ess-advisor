"""Validated exact-input data model and deterministic JSON serialization."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping


def parse_fraction(value: Any) -> Fraction:
    """Parse an integer, finite decimal, or rational JSON value exactly."""
    if isinstance(value, bool):
        raise TypeError("boolean is not a numeric coefficient")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("numeric value must be finite")
        return Fraction(str(value))
    if isinstance(value, str):
        try:
            return Fraction(value)
        except ValueError as exc:
            raise ValueError(f"invalid exact numeric value: {value!r}") from exc
    raise TypeError(f"unsupported numeric value: {type(value).__name__}")


def fraction_to_json(value: Fraction) -> str:
    value = parse_fraction(value)
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _sorted_endpoints(endpoints: tuple[str, str]) -> tuple[str, str]:
    if len(endpoints) != 2 or endpoints[0] == endpoints[1]:
        raise ValueError("a candidate must cover two distinct attributes")
    return tuple(sorted(endpoints))


@dataclass(frozen=True)
class Candidate:
    id: str
    table: str
    endpoints: tuple[str, str]
    maintenance_cost: Fraction

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("candidate id must be nonempty")
        if not self.table:
            raise ValueError("candidate table must be nonempty")
        object.__setattr__(self, "endpoints", _sorted_endpoints(self.endpoints))
        cost = parse_fraction(self.maintenance_cost)
        if cost < 0:
            raise ValueError("maintenance costs must be nonnegative")
        object.__setattr__(self, "maintenance_cost", cost)


@dataclass(frozen=True)
class Task:
    id: str
    table: str
    weight: Fraction
    alpha: Fraction
    eligible_candidates: tuple[str, ...]
    corrections: Mapping[str, Fraction]

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("task id must be nonempty")
        if not self.table:
            raise ValueError("task table must be nonempty")
        weight = parse_fraction(self.weight)
        if weight <= 0:
            raise ValueError("task weights must be strictly positive")
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "alpha", parse_fraction(self.alpha))
        eligible = tuple(sorted(set(self.eligible_candidates)))
        if len(eligible) != len(self.eligible_candidates):
            raise ValueError("task candidate references must be unique")
        object.__setattr__(self, "eligible_candidates", eligible)
        corrections = {
            str(key): parse_fraction(value)
            for key, value in self.corrections.items()
        }
        if set(corrections) != set(eligible):
            raise ValueError(
                "corrections must contain exactly the eligible candidate ids"
            )
        object.__setattr__(self, "corrections", corrections)


@dataclass(frozen=True)
class Instance:
    candidates: tuple[Candidate, ...]
    tasks: tuple[Task, ...]
    budget: Fraction

    def __post_init__(self) -> None:
        candidates = tuple(sorted(self.candidates, key=lambda item: item.id))
        tasks = tuple(sorted(self.tasks, key=lambda item: item.id))
        if len({item.id for item in candidates}) != len(candidates):
            raise ValueError("candidate ids must be globally unique")
        if len({item.id for item in tasks}) != len(tasks):
            raise ValueError("task ids must be globally unique")
        candidate_map = {item.id: item for item in candidates}
        for task in tasks:
            for candidate_id in task.eligible_candidates:
                candidate = candidate_map.get(candidate_id)
                if candidate is None:
                    raise ValueError(
                        f"task {task.id} references unknown candidate {candidate_id}"
                    )
                if candidate.table != task.table:
                    raise ValueError(
                        f"task {task.id} references candidate {candidate_id} "
                        f"from incompatible table {candidate.table}"
                    )
        budget = parse_fraction(self.budget)
        if budget < 0:
            raise ValueError("budget must be nonnegative")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "tasks", tasks)
        object.__setattr__(self, "budget", budget)

    @property
    def candidate_map(self) -> dict[str, Candidate]:
        return {item.id: item for item in self.candidates}

    @property
    def task_map(self) -> dict[str, Task]:
        return {item.id: item for item in self.tasks}

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(item.id for item in self.candidates)

    def maintenance_cost(self, selected: set[str] | frozenset[str]) -> Fraction:
        mapping = self.candidate_map
        unknown = set(selected) - set(mapping)
        if unknown:
            raise ValueError(f"unknown selected candidates: {sorted(unknown)}")
        return sum((mapping[item].maintenance_cost for item in selected), Fraction())

    def task_candidates(self, task: Task) -> tuple[Candidate, ...]:
        mapping = self.candidate_map
        return tuple(mapping[item] for item in task.eligible_candidates)


def instance_to_dict(instance: Instance) -> dict[str, Any]:
    return {
        "budget": fraction_to_json(instance.budget),
        "candidates": [
            {
                "id": candidate.id,
                "table": candidate.table,
                "endpoints": list(candidate.endpoints),
                "maintenance_cost": fraction_to_json(candidate.maintenance_cost),
            }
            for candidate in instance.candidates
        ],
        "tasks": [
            {
                "id": task.id,
                "table": task.table,
                "weight": fraction_to_json(task.weight),
                "alpha": fraction_to_json(task.alpha),
                "eligible_candidates": list(task.eligible_candidates),
                "corrections": {
                    key: fraction_to_json(task.corrections[key])
                    for key in sorted(task.corrections)
                },
            }
            for task in instance.tasks
        ],
    }


def instance_from_dict(data: Mapping[str, Any]) -> Instance:
    candidates = tuple(
        Candidate(
            id=item["id"],
            table=item["table"],
            endpoints=tuple(item["endpoints"]),
            maintenance_cost=parse_fraction(item["maintenance_cost"]),
        )
        for item in data["candidates"]
    )
    tasks = tuple(
        Task(
            id=item["id"],
            table=item["table"],
            weight=parse_fraction(item["weight"]),
            alpha=parse_fraction(item["alpha"]),
            eligible_candidates=tuple(item["eligible_candidates"]),
            corrections={
                key: parse_fraction(value)
                for key, value in item["corrections"].items()
            },
        )
        for item in data["tasks"]
    )
    return Instance(
        candidates=candidates,
        tasks=tasks,
        budget=parse_fraction(data["budget"]),
    )


def write_instance(path: str | os.PathLike[str], instance: Instance) -> None:
    """Write deterministic JSON atomically, replacing only the target file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    payload = json.dumps(
        instance_to_dict(instance),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    ) + "\n"
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, target)


def read_instance(path: str | os.PathLike[str]) -> Instance:
    return instance_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
