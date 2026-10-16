"""Deterministic synthetic PESS-ESS instance generation."""

from __future__ import annotations

import random
from fractions import Fraction

from .model import Candidate, Instance, Task


def generate_instance(
    seed: int,
    candidate_count: int,
    task_count: int,
    attribute_count: int = 5,
    eligibility_probability: float = 0.65,
    mixed_sign: bool = True,
    zero_cost: bool = False,
    tight_budget: bool = False,
    sign_mode: str | None = None,
) -> Instance:
    rng = random.Random(seed)
    if sign_mode is None:
        sign_mode = "mixed" if mixed_sign else "positive"
    if sign_mode not in {"mixed", "positive", "negative"}:
        raise ValueError(f"unknown sign mode: {sign_mode}")
    attributes = tuple(f"a{index}" for index in range(attribute_count))
    candidates: list[Candidate] = []
    for index in range(candidate_count):
        first, second = rng.sample(attributes, 2)
        if index > 0 and rng.random() < 0.2:
            prior = rng.choice(candidates)
            first, second = prior.endpoints
        cost = 0 if zero_cost and rng.random() < 0.25 else rng.randint(1, 3)
        candidates.append(
            Candidate(
                id=f"e{index:03d}",
                table="t0",
                endpoints=(first, second),
                maintenance_cost=Fraction(cost),
            )
        )
    tasks: list[Task] = []
    for task_index in range(task_count):
        eligible = tuple(
            candidate.id
            for candidate in candidates
            if rng.random() < eligibility_probability
        )
        if task_index == 0 and not eligible and candidates:
            eligible = (candidates[0].id,)
        if sign_mode == "positive":
            alpha = Fraction(rng.randint(0, 4))
        elif sign_mode == "negative":
            alpha = Fraction(-rng.randint(0, 4))
        else:
            alpha = Fraction(rng.randint(-4, 4))
        corrections: dict[str, Fraction] = {}
        for candidate_id in eligible:
            if sign_mode == "positive":
                numerator = rng.randint(0, 4)
            elif sign_mode == "negative":
                numerator = -rng.randint(0, 4)
            else:
                numerator = rng.randint(-4, 4)
            corrections[candidate_id] = Fraction(numerator, 2)
        tasks.append(
            Task(
                id=f"i{task_index:03d}",
                table="t0",
                weight=Fraction(rng.randint(1, 3)),
                alpha=alpha,
                eligible_candidates=eligible,
                corrections=corrections,
            )
        )
    total_cost = sum((candidate.maintenance_cost for candidate in candidates), Fraction())
    budget = (
        total_cost / 3
        if tight_budget
        else total_cost / 2
    )
    return Instance(tuple(candidates), tuple(tasks), budget)
