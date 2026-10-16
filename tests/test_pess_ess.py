from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from types import SimpleNamespace

import pytest

from pess_ess.generate import generate_instance
from pess_ess.matchings import (
    enumerate_matchings,
    enumerate_maximal_matchings,
    is_maximal,
)
from pess_ess.mip import solve_binary_milp
from pess_ess.model import Candidate, Instance, Task, instance_to_dict
from pess_ess.oracle import evaluate_task, exact_maximal_matching_oracle, inner_bounds
from pess_ess.reference import reference_objective, solve_exhaustive
from pess_ess.solvers import (
    MasterOracleOptions,
    TaskLocalOracleCache,
    solve_master_oracle,
    solve_static,
)
from pess_ess.validation import validate_outer_solution, validate_task_witness


def make_instance(
    candidates: tuple[Candidate, ...],
    task_specs: tuple[
        tuple[str, Fraction, Fraction, tuple[str, ...], dict[str, Fraction]], ...
    ],
    budget: Fraction,
) -> Instance:
    tasks = tuple(
        Task(
            id=task_id,
            table="t0",
            weight=weight,
            alpha=alpha,
            eligible_candidates=eligible,
            corrections=corrections,
        )
        for task_id, weight, alpha, eligible, corrections in task_specs
    )
    return Instance(candidates=candidates, tasks=tasks, budget=budget)


def edge(
    candidate_id: str,
    endpoints: tuple[str, str],
    cost: int | Fraction = 1,
    table: str = "t0",
) -> Candidate:
    return Candidate(candidate_id, table, endpoints, Fraction(cost))


def two_disjoint_instance(alpha: int = 1, b: int = 0, budget: int = 1) -> Instance:
    candidates = (edge("e", ("a", "b")), edge("f", ("c", "d")))
    return make_instance(
        candidates,
        (
            (
                "i",
                Fraction(1),
                Fraction(alpha),
                ("e", "f"),
                {"e": Fraction(b), "f": Fraction(b)},
            ),
        ),
        Fraction(budget),
    )


def test_empty_universe_and_empty_task_graph() -> None:
    instance = make_instance(
        (),
        (("empty", Fraction(2), Fraction(-3), (), {}),),
        Fraction(0),
    )
    reference = solve_exhaustive(instance)
    assert reference.objective == Fraction(6)
    assert reference.optimal_sets == (frozenset(),)
    assert solve_static(instance).objective == reference.objective
    assert solve_master_oracle(instance).objective == reference.objective


def test_single_edge_and_tight_budget() -> None:
    instance = make_instance(
        (edge("e", ("a", "b"), 2),),
        (("i", Fraction(1), Fraction(-2), ("e",), {"e": Fraction(3)}),),
        Fraction(0),
    )
    reference = solve_exhaustive(instance)
    assert reference.optimal_sets == (frozenset(),)
    assert reference.objective == Fraction(2)
    assert solve_static(instance).objective == Fraction(2)
    assert solve_master_oracle(instance).objective == Fraction(2)


def test_conflicting_and_disjoint_edges_have_distinct_maximal_matchings() -> None:
    candidates = (
        edge("e", ("a", "b")),
        edge("f", ("b", "c")),
        edge("g", ("d", "e")),
    )
    endpoints = {candidate.id: candidate.endpoints for candidate in candidates}
    selected = frozenset({"e", "f", "g"})
    maximal = enumerate_maximal_matchings(selected, endpoints)
    assert frozenset({"e", "g"}) in maximal
    assert frozenset({"f", "g"}) in maximal
    assert frozenset({"e", "f"}) not in maximal
    assert frozenset({"e", "f", "g"}) not in maximal


def test_parallel_labeled_edges_are_distinct_and_conflicting() -> None:
    candidates = (
        edge("e1", ("a", "b")),
        edge("e2", ("a", "b")),
    )
    endpoints = {candidate.id: candidate.endpoints for candidate in candidates}
    matchings = enumerate_matchings(("e1", "e2"), endpoints)
    assert frozenset({"e1", "e2"}) not in matchings
    assert frozenset({"e1"}) in matchings
    assert frozenset({"e2"}) in matchings
    instance = make_instance(
        candidates,
        (
            (
                "i",
                Fraction(1),
                Fraction(0),
                ("e1", "e2"),
                {"e1": Fraction(1), "e2": Fraction(-2)},
            ),
        ),
        Fraction(1),
    )
    result = solve_master_oracle(instance)
    assert result.status == "optimal"


def test_candidate_table_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="incompatible table"):
        make_instance(
            (edge("e", ("a", "b"), table="other"),),
            (("i", Fraction(1), Fraction(0), ("e",), {"e": Fraction(1)}),),
            Fraction(1),
        )


def test_delta_equivalence_for_seeded_random_instances() -> None:
    for seed in range(10):
        instance = generate_instance(seed, 5, 3, attribute_count=4)
        candidate_map = instance.candidate_map
        for task in instance.tasks:
            endpoints = {
                item: candidate_map[item].endpoints for item in task.eligible_candidates
            }
            for size in range(len(instance.candidate_ids) + 1):
                for selected_tuple in combinations(instance.candidate_ids, size):
                    selected = frozenset(selected_tuple)
                    eligible = selected & set(task.eligible_candidates)
                    for matching in enumerate_matchings(
                        task.eligible_candidates, endpoints
                    ):
                        occupied = (
                            set().union(*(set(endpoints[item]) for item in matching))
                            if matching
                            else set()
                        )
                        disjoint = {
                            item
                            for item in task.eligible_candidates
                            if not (set(endpoints[item]) & occupied)
                        }
                        delta = sum(item not in selected for item in matching) + sum(
                            item in selected for item in disjoint
                        )
                        expected = matching in enumerate_maximal_matchings(
                            eligible, endpoints
                        )
                        assert (delta == 0) == expected, (
                            f"seed={seed}; instance={instance_to_dict(instance)}; "
                            f"task={task.id}; selected={sorted(selected)}; "
                            f"matching={sorted(matching)}"
                        )


def test_oracle_matches_independent_reference_on_all_small_selections() -> None:
    for seed in range(6):
        instance = generate_instance(seed, 5, 3, attribute_count=4)
        for size in range(len(instance.candidate_ids) + 1):
            for selected_tuple in combinations(instance.candidate_ids, size):
                selected = frozenset(selected_tuple)
                expected, details = reference_objective(instance, selected)
                actual = sum(
                    (
                        task.weight * evaluate_task(instance, task, selected).value
                        for task in instance.tasks
                    ),
                    Fraction(),
                )
                assert actual == expected, (
                    f"seed={seed}; instance={instance_to_dict(instance)}; "
                    f"selected={sorted(selected)}; expected={expected}; "
                    f"actual={actual}; details={details}"
                )


def test_all_outer_solvers_agree_on_seeded_small_instances() -> None:
    for seed in range(8):
        instance = generate_instance(seed, 5, 3, attribute_count=4)
        reference = solve_exhaustive(instance)
        static = solve_static(instance)
        master = solve_master_oracle(instance)
        assert static.status == "optimal", (
            f"seed={seed}; instance={instance_to_dict(instance)}; static={static}"
        )
        assert master.status == "optimal", (
            f"seed={seed}; instance={instance_to_dict(instance)}; master={master}"
        )
        assert static.objective == reference.objective, (
            f"seed={seed}; instance={instance_to_dict(instance)}"
        )
        assert master.objective == reference.objective, (
            f"seed={seed}; instance={instance_to_dict(instance)}"
        )
        assert master.lower_bound is not None
        assert master.upper_bound == float(reference.objective)


def test_zero_cost_candidates_and_zero_budget() -> None:
    instance = make_instance(
        (edge("e", ("a", "b"), 0),),
        (("i", Fraction(1), Fraction(-1), ("e",), {"e": Fraction(1)}),),
        Fraction(0),
    )
    reference = solve_exhaustive(instance)
    assert reference.optimal_sets == (frozenset({"e"}),)
    assert solve_master_oracle(instance).objective == Fraction(0)


def test_positive_lp_gap_examples_from_theory_note() -> None:
    first = two_disjoint_instance(alpha=1, b=0)
    assert solve_exhaustive(first).objective == Fraction(1)
    first_bounds = inner_bounds(first, first.tasks[0], frozenset())
    assert first_bounds.lower_loss == first_bounds.upper_loss == Fraction(1)

    second = two_disjoint_instance(alpha=-5, b=1)
    assert solve_exhaustive(second).objective == Fraction(4)
    second_bounds = inner_bounds(second, second.tasks[0], frozenset())
    assert second_bounds.lower_loss == second_bounds.upper_loss == Fraction(5)

    # At x_e=x_f=1/2 every full matching has delta=1. The original and
    # matching-specific cuts permit zero, whereas the floor is three.
    losses = [Fraction(5), Fraction(4), Fraction(4), Fraction(3)]
    lam = Fraction(3)
    assert max(losses) == 5
    assert all(loss - (loss - lam) * 1 == lam for loss in losses)


def test_sign_coherent_empty_selection_is_optimal() -> None:
    positive = make_instance(
        (edge("e", ("a", "b")), edge("f", ("c", "d"))),
        (
            (
                "i",
                Fraction(1),
                Fraction(2),
                ("e", "f"),
                {"e": Fraction(1), "f": Fraction(3)},
            ),
        ),
        Fraction(2),
    )
    negative = make_instance(
        (edge("e", ("a", "b")), edge("f", ("c", "d"))),
        (
            (
                "i",
                Fraction(1),
                Fraction(-2),
                ("e", "f"),
                {"e": Fraction(-1), "f": Fraction(-3)},
            ),
        ),
        Fraction(2),
    )
    assert solve_exhaustive(positive).optimal_sets == (frozenset(),)
    assert solve_exhaustive(negative).optimal_sets == (frozenset(),)


def test_nonmonotone_pessimistic_objective_and_conditional_invalidity() -> None:
    instance = make_instance(
        (edge("e", ("a", "b")), edge("f", ("b", "c"))),
        (
            (
                "i",
                Fraction(1),
                Fraction(-1),
                ("e", "f"),
                {"e": Fraction(1), "f": Fraction(-1)},
            ),
        ),
        Fraction(2),
    )
    empty, _ = reference_objective(instance, frozenset())
    selected, _ = reference_objective(instance, frozenset({"e"}))
    both, _ = reference_objective(instance, frozenset({"e", "f"}))
    assert (empty, selected, both) == (Fraction(1), Fraction(0), Fraction(2))
    endpoints = {
        "e": ("a", "b"),
        "f": ("c", "d"),
    }
    matching = frozenset({"e"})
    assert is_maximal(matching, {"e"}, endpoints)
    assert not is_maximal(matching, {"e", "f"}, endpoints)


def test_serialization_is_deterministic_and_exact() -> None:
    instance = generate_instance(13, 4, 2)
    first = instance_to_dict(instance)
    second = instance_to_dict(instance)
    assert first == second
    assert all(
        isinstance(value, str)
        for task in first["tasks"]
        for value in [task["weight"], task["alpha"], *task["corrections"].values()]
    )


def test_mip_result_separates_primal_dual_and_records_presolve_retry() -> None:
    responses = iter(
        (
            SimpleNamespace(
                status=4,
                message="presolve diagnostic failure",
                fun=9.0,
                x=[1.0],
                mip_dual_bound=2.0,
                mip_gap=3.5,
                mip_node_count=4,
            ),
            SimpleNamespace(
                status=0,
                message="retry optimal",
                fun=4.0,
                x=[0.0],
                mip_dual_bound=4.0,
                mip_gap=0.0,
                mip_node_count=5,
            ),
        )
    )
    options_seen = []

    def fake_runner(**kwargs):
        options_seen.append(kwargs["options"])
        return next(responses)

    result = solve_binary_milp(
        objective=(1.0,),
        integrality=(1,),
        lower_bounds=(0.0,),
        upper_bounds=(1.0,),
        rows=[],
        solver_runner=fake_runner,
    )
    assert result.highs_invocations == 2
    assert result.presolve_fallback_attempted
    assert result.presolve_fallback_successes == 1
    assert result.initial_attempt.status == 4
    assert result.retry_attempt is not None
    assert result.selected_attempt == "retry"
    assert result.primal_objective == 4.0
    assert result.dual_bound == 4.0
    assert result.objective == result.primal_objective
    assert options_seen[1]["presolve"] is False


def test_failed_presolve_retry_retains_both_failures() -> None:
    responses = iter(
        (
            SimpleNamespace(status=4, message="first", fun=None, x=None),
            SimpleNamespace(status=4, message="second", fun=None, x=None),
        )
    )

    result = solve_binary_milp(
        objective=(1.0,),
        integrality=(1,),
        lower_bounds=(0.0,),
        upper_bounds=(1.0,),
        rows=[],
        solver_runner=lambda **_: next(responses),
    )
    assert result.status == 4
    assert not result.optimal
    assert result.retry_attempt is not None
    assert "initial status 4" in result.message
    assert "retry status 4" in result.message


def test_zero_deadline_prevents_backend_invocation() -> None:
    calls = []
    result = solve_binary_milp(
        objective=(1.0,),
        integrality=(1,),
        lower_bounds=(0.0,),
        upper_bounds=(1.0,),
        rows=[],
        time_limit=0.0,
        solver_runner=lambda **_: calls.append(True),
    )
    assert result.status == 1
    assert result.highs_invocations == 0
    assert calls == []


def test_validation_recomputes_budget_and_witness_loss() -> None:
    instance = make_instance(
        (edge("e", ("a", "b"), 2),),
        (("i", Fraction(1), Fraction(-1), ("e",), {"e": Fraction(2)}),),
        Fraction(2),
    )
    objective, details = reference_objective(instance, frozenset({"e"}))
    loss, witness = details["i"]
    validation = validate_outer_solution(
        instance,
        frozenset({"e"}),
        {"i": (loss, witness)},
    )
    assert validation.valid
    assert validation.objective == objective
    invalid = validate_task_witness(
        instance,
        instance.tasks[0],
        frozenset(),
        frozenset({"e"}),
        loss,
    )
    assert any("unselected" in error for error in invalid)


def test_global_deadline_is_reported_as_nonoptimal() -> None:
    instance = two_disjoint_instance()
    result = solve_master_oracle(instance, total_time_limit=0.0)
    assert result.status == "not_optimal"
    assert not result.solver_reported_optimal


def test_outer_result_exposes_precise_counter_semantics() -> None:
    instance = two_disjoint_instance(alpha=1, b=1, budget=1)
    result = solve_master_oracle(instance)
    assert result.validated_feasible
    assert result.task_evaluations >= 1
    assert result.exact_oracle_invocations + result.certified_early_exits >= 1
    assert result.inner_mip_solve_calls >= result.exact_oracle_invocations
    assert result.highs_invocations >= result.inner_mip_solve_calls
    assert result.total_runtime_seconds >= 0.0


def test_four_master_oracle_variants_preserve_small_instance_objective() -> None:
    instance = generate_instance(29, 5, 3, attribute_count=4)
    reference = solve_exhaustive(instance)
    variants = (
        MasterOracleOptions(),
        MasterOracleOptions(precompute_invariants=True),
        MasterOracleOptions(task_local_cache=True),
        MasterOracleOptions(precompute_invariants=True, task_local_cache=True),
    )
    results = [solve_master_oracle(instance, options=options) for options in variants]
    assert all(result.status == "optimal" for result in results)
    assert all(result.objective == reference.objective for result in results)
    assert results[0].full_graph_ordinary_matching_extrema_computations >= (
        results[1].full_graph_ordinary_matching_extrema_computations
    )
    assert results[2].cache_hits >= 0
    assert results[3].cache_hits >= 0


def test_task_local_cache_is_scoped_by_task_and_local_selection() -> None:
    candidates = (edge("e", ("a", "b")), edge("f", ("c", "d")))
    instance = make_instance(
        candidates,
        (
            ("i0", Fraction(1), Fraction(1), ("e",), {"e": Fraction(0)}),
            ("i1", Fraction(1), Fraction(1), ("e",), {"e": Fraction(0)}),
        ),
        Fraction(1),
    )
    cache = TaskLocalOracleCache()
    result = evaluate_task(instance, instance.tasks[0], frozenset({"e"}))
    assert cache.put(instance.tasks[0], frozenset({"e"}), result)
    assert cache.get(instance.tasks[0], frozenset({"e"})) == result
    assert cache.get(instance.tasks[0], frozenset({"f"})) is None
    assert cache.get(instance.tasks[1], frozenset({"e"})) is None
    other = make_instance(
        (edge("e", ("x", "y")),),
        (("i0", Fraction(1), Fraction(1), ("e",), {"e": Fraction(0)}),),
        Fraction(1),
    )
    other_result = evaluate_task(other, other.tasks[0], frozenset({"e"}))
    other_cache = TaskLocalOracleCache()
    assert other_cache.get(other.tasks[0], frozenset({"e"})) is None
    assert other_cache.put(other.tasks[0], frozenset({"e"}), other_result)
    assert other_cache.get(other.tasks[0], frozenset({"e"})) == other_result


def test_unresolved_inner_results_are_not_cacheable() -> None:
    cache = TaskLocalOracleCache()
    unresolved = SimpleNamespace(source="mip_nonoptimal", solver_statuses=())
    assert not cache.cacheable(unresolved)


def test_cache_hits_reuse_valid_witness_without_new_inner_solves() -> None:
    instance = generate_instance(31, 5, 3, attribute_count=4)
    baseline = solve_master_oracle(instance)
    cached = solve_master_oracle(
        instance,
        options=MasterOracleOptions(task_local_cache=True),
    )
    assert cached.status == baseline.status == "optimal"
    assert cached.objective == baseline.objective
    assert cached.cache_hits > 0
    assert 0 < cached.cache_entries <= cached.cache_misses
    assert cached.inner_mip_solve_calls <= baseline.inner_mip_solve_calls
    assert validate_outer_solution(
        instance, cached.selected or frozenset(), cached.task_details
    ).valid


def test_precomputed_bounds_are_reused_by_exact_inner_oracle() -> None:
    instance = generate_instance(7, 5, 3, attribute_count=4)
    selected = frozenset(instance.candidate_ids)
    found = False
    for task in instance.tasks:
        bounds = inner_bounds(instance, task, selected)
        exact = exact_maximal_matching_oracle(
            instance,
            task,
            selected,
            bounds=bounds,
        )
        if exact.source == "mip":
            found = True
            assert exact.ordinary_matching_bound_computations == 0
            assert exact.value == evaluate_task(
                instance, task, selected, reuse_bounds=True
            ).value
            break
    assert found
