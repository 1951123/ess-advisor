from __future__ import annotations

import json
from fractions import Fraction
from io import StringIO

import pytest

from pess_ess import Candidate, Instance, Task, solve, solve_payload
from pess_ess.mip import MIPAttempt, MIPResult
from pess_ess.model import instance_to_dict
from pess_ess.reference import solve_exhaustive
from pess_ess.solvers import solve_master_oracle


def adversarial_instance() -> Instance:
    candidates = (
        Candidate("ab-1", "t", ("a", "b"), Fraction(0)),
        Candidate("ab-2", "t", ("a", "b"), Fraction(1)),
        Candidate("bc", "t", ("b", "c"), Fraction(1)),
        Candidate("cd", "t", ("c", "d"), Fraction(1)),
    )
    task = Task(
        "q1",
        "t",
        Fraction(2),
        Fraction(1, 3),
        ("ab-1", "ab-2", "bc", "cd"),
        {
            "ab-1": Fraction(-2),
            "ab-2": Fraction(1, 2),
            "bc": Fraction(1),
            "cd": Fraction(-1, 2),
        },
    )
    return Instance(candidates, (task,), Fraction(2))


def test_both_backends_match_fraction_reference_on_mixed_sign_parallel_case() -> None:
    instance = adversarial_instance()
    reference = solve_exhaustive(instance)
    for backend in ("master_oracle", "static"):
        result = solve(instance, objective="pess", backend=backend)
        assert result.status == "optimal"
        assert result.validated_feasible
        assert result.objective == reference.objective


def test_json_schema_round_trip_and_provenance() -> None:
    request = {
        "schema_version": "1",
        "objective": "pess",
        "backend": "master_oracle",
        "limits": {"time_seconds": 15, "total_time_seconds": 30},
        "options": {"task_local_cache": True},
        "input_provenance": {
            "model_version": "forest10-positive-preclamp-v1",
            "advisor_source_commit": "a" * 40,
        },
        "instance": instance_to_dict(adversarial_instance()),
    }
    response = solve_payload(json.loads(json.dumps(request)))
    assert response["schema_version"] == "1"
    assert response["status"] == "optimal"
    assert response["feasibility"] == "feasible"
    assert response["incumbent"]["recomputed_objective"] is not None
    assert response["input_provenance"] == request["input_provenance"]
    assert response["provenance"]["scipy_version"]
    assert response["optimality"]["kind"] == "floating_point_milp_status"


def test_json_cli_reads_stdin_and_writes_one_result(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from pess_ess.cli import main

    request = {
        "schema_version": "1",
        "objective": "pess",
        "backend": "master_oracle",
        "instance": instance_to_dict(adversarial_instance()),
    }
    monkeypatch.setattr("sys.argv", ["pess-ess-solve"])
    monkeypatch.setattr("sys.stdin", StringIO(json.dumps(request)))
    assert main() == 0
    response = json.loads(capsys.readouterr().out)
    assert response["schema_version"] == "1"
    assert response["feasibility"] == "feasible"


def test_only_pess_and_supported_backend_limits_are_accepted() -> None:
    instance = adversarial_instance()
    with pytest.raises(ValueError, match="only objective='pess'"):
        solve(instance, objective="joint_robust")
    with pytest.raises(ValueError, match="inner_time_limit"):
        solve(instance, backend="static", inner_time_limit=1)
    with pytest.raises(ValueError, match="nonnegative"):
        solve(instance, time_limit=-1)
    with pytest.raises(ValueError, match="schema_version"):
        solve_payload({"schema_version": "2"})


def _time_limit_incumbent(selection: tuple[float, ...]) -> MIPResult:
    attempt = MIPAttempt(
        status=1,
        message="time limit reached with incumbent",
        primal_objective=0.0,
        dual_bound=0.0,
        mip_gap=None,
        mip_node_count=1,
        values=selection,
        duration_seconds=0.01,
        invocation_performed=True,
        presolve=None,
        incumbent_returned=True,
        optimality_reported=False,
    )
    return MIPResult(
        status=1,
        message=attempt.message,
        primal_objective=attempt.primal_objective,
        dual_bound=attempt.dual_bound,
        values=attempt.values,
        mip_gap=attempt.mip_gap,
        mip_node_count=attempt.mip_node_count,
        initial_attempt=attempt,
        retry_attempt=None,
        selected_attempt="initial",
        total_duration_seconds=attempt.duration_seconds,
        highs_invocations=1,
        presolve_fallback_attempted=False,
        presolve_fallback_successes=0,
    )


def test_time_limited_outer_incumbent_is_accepted_only_after_independent_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = Instance(
        (Candidate("e", "t", ("a", "b"), Fraction(1)),),
        (
            Task(
                "q",
                "t",
                Fraction(1),
                Fraction(-1),
                ("e",),
                {"e": Fraction(1)},
            ),
        ),
        Fraction(1),
    )
    monkeypatch.setattr(
        "pess_ess.solvers_v2._solve_conditional_master",
        lambda *args, **kwargs: (_time_limit_incumbent((1.0, 0.0)), {}, 0.0),
    )
    result = solve_master_oracle(instance)
    assert result.status == "feasible_unproven"
    assert result.selected == frozenset({"e"})
    assert result.objective == 0
    assert result.validated_feasible
    assert not result.solver_reported_optimal


def test_invalid_fractional_master_incumbent_is_not_returned_as_feasible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = Instance(
        (Candidate("e", "t", ("a", "b"), Fraction(1)),),
        (
            Task(
                "q",
                "t",
                Fraction(1),
                Fraction(-1),
                ("e",),
                {"e": Fraction(1)},
            ),
        ),
        Fraction(1),
    )
    monkeypatch.setattr(
        "pess_ess.solvers_v2._solve_conditional_master",
        lambda *args, **kwargs: (_time_limit_incumbent((0.5, 0.0)), {}, 0.0),
    )
    result = solve_master_oracle(instance)
    assert result.status == "not_optimal"
    assert result.selected is None
    assert not result.validated_feasible


def test_time_limited_static_incumbent_is_independently_recomputed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = Instance(
        (Candidate("e", "t", ("a", "b"), Fraction(1)),),
        (
            Task(
                "q",
                "t",
                Fraction(1),
                Fraction(-1),
                ("e",),
                {"e": Fraction(1)},
            ),
        ),
        Fraction(1),
    )
    monkeypatch.setattr(
        "pess_ess.solvers_v2._solve_conditional_master",
        lambda *args, **kwargs: (_time_limit_incumbent((1.0, 0.0)), {}, 0.0),
    )
    result = solve(instance, backend="static")
    assert result.status == "feasible_unproven"
    assert result.objective == 0
    assert result.validated_feasible
    assert not result.solver_reported_optimal
