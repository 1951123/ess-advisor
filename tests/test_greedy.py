from __future__ import annotations

from fractions import Fraction

import pytest

from pess_ess import Candidate, Instance, Task, solve, solve_payload
from pess_ess.api import result_to_dict
from pess_ess.model import instance_to_dict
from pess_ess.reference import (
    reference_nominal_objective,
    reference_objective,
    solve_exhaustive,
)


def disjoint_instance() -> Instance:
    return Instance(
        (
            Candidate("z-resource", "t", ("e", "f"), Fraction(1)),
            Candidate("a-resource", "t", ("a", "b"), Fraction(1)),
            Candidate("m-resource", "t", ("c", "d"), Fraction(1)),
        ),
        (
            Task(
                "q",
                "t",
                Fraction(1),
                Fraction(-5),
                ("z-resource", "a-resource", "m-resource"),
                {"z-resource": Fraction(2), "a-resource": Fraction(3),
                 "m-resource": Fraction(1)},
            ),
        ),
        Fraction(3),
    )


def test_common_backend_runs_nominal_and_pess_with_fraction_steps() -> None:
    instance = disjoint_instance()
    priority = ("m-resource", "z-resource", "a-resource")

    pess = solve(instance, objective="pess", backend="add_only_greedy")
    nominal = solve(
        instance,
        objective="nominal",
        backend="add_only_greedy",
        priority_order=priority,
    )

    for result in (pess, nominal):
        assert result.status == "heuristic_completed"
        assert result.termination_reason == "no_strict_improvement"
        assert result.addition_order == ("a-resource", "z-resource")
        assert result.selected == frozenset({"a-resource", "z-resource"})
        assert result.objective == 0
        assert result.maintenance_cost == 2
        assert result.unused_budget == 1
        assert [step.improvement for step in result.steps] == [3, 2]
        assert all(step.improvement > 0 for step in result.steps)
        assert result.fixed_set_evaluation_calls == 7
        assert result.completed_fixed_set_evaluations == 7

    assert pess.inner_matching_subsets_examined == 23
    assert pess.inner_maximal_matchings_scored == 7
    assert nominal.inner_matching_subsets_examined == 0
    assert nominal.inner_maximal_matchings_scored == 0

    for objective, order in (("pess", None), ("nominal", list(priority))):
        request = {
            "schema_version": "1",
            "objective": objective,
            "backend": "add_only_greedy",
            "instance": instance_to_dict(instance),
        }
        if order is not None:
            request["priority_order"] = order
        response = solve_payload(request)
        assert response["status"] == "heuristic_completed"
        assert response["optimality"]["solver_reported_optimal"] is False
        assert response["optimality"]["outer_global_optimum_fraction_certified"] is False
        assert response["incumbent"]["selected"] == ["a-resource", "z-resource"]
        assert response["incumbent"]["objective_value"] == "0"
        assert response["greedy"]["addition_order"] == [
            "a-resource", "z-resource"
        ]
        assert response["greedy"]["tie_break_rule"] == "ascending_candidate_id"

    nominal_dict = result_to_dict(
        nominal,
        backend="add_only_greedy",
        objective="nominal",
        priority_order=priority,
    )
    assert nominal_dict["policy"]["priority_order"] == list(priority)


def test_each_pess_candidate_evaluation_matches_exact_reference(monkeypatch) -> None:
    import pess_ess.greedy as greedy_module

    instance = disjoint_instance()
    original = greedy_module.reference_objective
    observed: list[tuple[frozenset[str], Fraction]] = []

    def recording_reference(instance, selected, **kwargs):
        result = original(instance, selected, **kwargs)
        observed.append((selected, result[0]))
        return result

    monkeypatch.setattr(greedy_module, "reference_objective", recording_reference)
    result = solve(instance, objective="pess", backend="add_only_greedy")

    assert len(observed) == result.fixed_set_evaluation_calls
    for selected, value in observed:
        assert value == reference_objective(instance, selected)[0]
    assert observed[0] == (frozenset(), Fraction(5))


def test_candidate_id_ties_use_canonical_order_independent_of_input_order() -> None:
    def build(order: tuple[str, str]) -> Instance:
        by_id = {
            "z-resource": Candidate("z-resource", "t", ("a", "b"), Fraction(0)),
            "a-resource": Candidate("a-resource", "t", ("a", "b"), Fraction(0)),
        }
        return Instance(
            tuple(by_id[item] for item in order),
            (
                Task(
                    "q",
                    "t",
                    Fraction(1),
                    Fraction(-2),
                    order,
                    {item: Fraction(2) for item in order},
                ),
            ),
            Fraction(0),
        )

    forward = solve(
        build(("z-resource", "a-resource")),
        objective="pess",
        backend="add_only_greedy",
    )
    reverse = solve(
        build(("a-resource", "z-resource")),
        objective="pess",
        backend="add_only_greedy",
    )
    assert forward.addition_order == reverse.addition_order == ("a-resource",)
    assert forward.steps[0].improvement == reverse.steps[0].improvement == 2


@pytest.mark.parametrize(
    ("endpoints", "corrections", "alpha", "expected_witnesses", "expected_loss"),
    [
        (
            (("a", "b"), ("c", "d")),
            (Fraction(2), Fraction(-1)),
            Fraction(-1),
            frozenset({"e1", "e2"}),
            Fraction(0),
        ),
        (
            (("a", "b"), ("b", "c"), ("c", "d")),
            (Fraction(2), Fraction(-3), Fraction(1)),
            Fraction(-1),
            frozenset({"e2"}),
            Fraction(4),
        ),
        (
            (("x", "a"), ("x", "b"), ("x", "c")),
            (Fraction(2), Fraction(-3), Fraction(1)),
            Fraction(-1),
            frozenset({"e2"}),
            Fraction(4),
        ),
        (
            (("a", "b"), ("a", "b")),
            (Fraction(2), Fraction(-3)),
            Fraction(-1),
            frozenset({"e2"}),
            Fraction(4),
        ),
    ],
    ids=("disjoint", "path", "star", "parallel_labeled_edges"),
)
def test_exact_pess_reference_covers_matching_shapes_and_mixed_signs(
    endpoints, corrections, alpha, expected_witnesses, expected_loss
) -> None:
    from pess_ess.reference import reference_task_loss

    ids = tuple(f"e{index + 1}" for index in range(len(endpoints)))
    instance = Instance(
        tuple(
            Candidate(item, "t", edge, Fraction(0 if index == 0 else 1))
            for index, (item, edge) in enumerate(zip(ids, endpoints))
        ),
        (
            Task(
                "q",
                "t",
                Fraction(1),
                alpha,
                ids,
                dict(zip(ids, corrections)),
            ),
        ),
        Fraction(10),
    )
    value, witness = reference_task_loss(instance, instance.tasks[0], frozenset(ids))
    assert value == expected_loss
    assert witness == expected_witnesses


def test_pess_can_worsen_after_overlap_and_greedy_keeps_best_set() -> None:
    instance = Instance(
        (
            Candidate("good", "t", ("a", "b"), Fraction(1)),
            Candidate("bad", "t", ("a", "b"), Fraction(1)),
        ),
        (
            Task(
                "q",
                "t",
                Fraction(1),
                Fraction(-1),
                ("good", "bad"),
                {"good": Fraction(1), "bad": Fraction(3)},
            ),
        ),
        Fraction(2),
    )
    before, _ = reference_objective(instance, frozenset({"good"}))
    after, _ = reference_objective(instance, frozenset({"good", "bad"}))
    result = solve(instance, objective="pess", backend="add_only_greedy")

    assert before == 0
    assert after == 2
    assert result.selected == frozenset({"good"})
    assert result.objective == before
    assert result.termination_reason == "no_strict_improvement"


def test_plateau_stops_even_when_two_additions_together_improve() -> None:
    instance = Instance(
        (
            Candidate("e1", "t", ("a", "b"), Fraction(1)),
            Candidate("e2", "t", ("c", "d"), Fraction(1)),
        ),
        (
            Task(
                "q",
                "t",
                Fraction(1),
                Fraction(1),
                ("e1", "e2"),
                {"e1": Fraction(2), "e2": Fraction(-3)},
            ),
        ),
        Fraction(2),
    )
    result = solve(instance, objective="pess", backend="add_only_greedy")
    exact = solve_exhaustive(instance)

    assert result.selected == frozenset()
    assert result.objective == 1
    assert result.termination_reason == "no_strict_improvement"
    assert exact.objective == 0
    assert result.objective - exact.objective == 1


def test_zero_cost_resource_is_added_at_zero_budget_then_stops_on_feasibility() -> None:
    instance = Instance(
        (
            Candidate("free", "t", ("a", "b"), Fraction(0)),
            Candidate("paid", "t", ("c", "d"), Fraction(1)),
        ),
        (
            Task(
                "q",
                "t",
                Fraction(1),
                Fraction(1),
                ("free", "paid"),
                {"free": Fraction(-1), "paid": Fraction(0)},
            ),
        ),
        Fraction(0),
    )
    result = solve(instance, objective="pess", backend="add_only_greedy")

    assert result.addition_order == ("free",)
    assert result.selected == frozenset({"free"})
    assert result.objective == 0
    assert result.maintenance_cost == 0
    assert result.unused_budget == 0
    assert result.termination_reason == "no_feasible_candidate"


def test_max_evaluation_and_time_limits_return_only_completed_steps() -> None:
    instance = disjoint_instance()

    no_initial_call = solve(
        instance,
        objective="pess",
        backend="add_only_greedy",
        max_evaluations=0,
    )
    after_baseline = solve(
        instance,
        objective="pess",
        backend="add_only_greedy",
        max_evaluations=1,
    )
    no_time = solve(
        instance,
        objective="pess",
        backend="add_only_greedy",
        total_time_limit=0,
    )

    assert no_initial_call.status == "heuristic_incomplete"
    assert no_initial_call.objective is None
    assert no_initial_call.fixed_set_evaluation_calls == 0
    assert no_initial_call.termination_detail == "max_evaluations"
    assert after_baseline.objective == 5
    assert after_baseline.selected == frozenset()
    assert after_baseline.steps == ()
    assert after_baseline.fixed_set_evaluation_calls == 1
    assert after_baseline.termination_detail == "max_evaluations"
    assert no_time.status == "heuristic_incomplete"
    assert no_time.objective is None
    assert no_time.termination_detail == "total_time_limit"


def test_inner_work_cap_preserves_only_completed_rounds() -> None:
    instance = Instance(
        (
            Candidate("a", "t", ("a", "b"), Fraction(1)),
            Candidate("b", "t", ("c", "d"), Fraction(1)),
        ),
        (
            Task(
                "q",
                "t",
                Fraction(1),
                Fraction(-2),
                ("a", "b"),
                {"a": Fraction(2), "b": Fraction(1)},
            ),
        ),
        Fraction(2),
    )
    result = solve(
        instance,
        objective="pess",
        backend="add_only_greedy",
        max_inner_matching_subsets=5,
    )

    assert result.status == "heuristic_incomplete"
    assert result.termination_reason == "evaluation_incomplete"
    assert result.termination_detail == "max_inner_matching_subsets"
    assert result.selected == frozenset({"a"})
    assert result.objective == 0
    assert result.addition_order == ("a",)
    assert result.inner_matching_subsets_examined == 5


def test_incomplete_inner_evaluation_does_not_commit_partial_round(monkeypatch) -> None:
    import pess_ess.greedy as greedy_module

    instance = Instance(
        (
            Candidate("a", "t", ("a", "b"), Fraction(1)),
            Candidate("b", "t", ("c", "d"), Fraction(1)),
        ),
        (
            Task(
                "q",
                "t",
                Fraction(1),
                Fraction(-2),
                ("a", "b"),
                {"a": Fraction(2), "b": Fraction(1)},
            ),
        ),
        Fraction(2),
    )
    original = greedy_module.reference_objective
    calls = 0

    def fail_during_first_round(instance, selected, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise TimeoutError("injected incomplete inner evaluation")
        return original(instance, selected, **kwargs)

    monkeypatch.setattr(greedy_module, "reference_objective", fail_during_first_round)
    result = solve(instance, objective="pess", backend="add_only_greedy")

    assert result.status == "heuristic_incomplete"
    assert result.termination_detail == "inner_evaluation_incomplete"
    assert result.selected == frozenset()
    assert result.steps == ()
    assert result.objective == 2
    assert result.fixed_set_evaluation_calls == 3
    assert result.completed_fixed_set_evaluations == 2


def test_nominal_fixed_set_evaluations_use_the_supplied_global_priority() -> None:
    instance = disjoint_instance()
    order = ("m-resource", "a-resource", "z-resource")
    result = solve(
        instance,
        objective="nominal",
        backend="add_only_greedy",
        priority_order=order,
    )
    assert result.objective == reference_nominal_objective(
        instance, result.selected, order
    )[0]
    assert result.priority_order == order
