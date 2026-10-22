"""Small PESS-only callable and JSON contract over the existing solvers."""

from __future__ import annotations

import math
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Literal

import scipy

from .model import Instance, instance_from_dict
from .solvers import OuterResult, solve_master_oracle, solve_static

Backend = Literal["master_oracle", "static"]
SCHEMA_VERSION = "1"


def solve(
    instance: Instance,
    *,
    objective: str = "pess",
    backend: Backend = "master_oracle",
    time_limit: float | None = None,
    inner_time_limit: float | None = None,
    total_time_limit: float | None = None,
    precompute_invariants: bool = False,
    task_local_cache: bool = False,
) -> OuterResult:
    """Solve the structured task-wise PESS model using a conventional MILP.

    The default master-oracle backend does not enumerate outer resource sets.
    The static backend enumerates complete matching scenarios and is intended
    for instances where that family remains tractable.
    """
    if objective != "pess":
        raise ValueError("only objective='pess' is supported by this API")
    if backend not in {"master_oracle", "static"}:
        raise ValueError("backend must be 'master_oracle' or 'static'")
    for name, limit in (
        ("time_limit", time_limit),
        ("inner_time_limit", inner_time_limit),
        ("total_time_limit", total_time_limit),
    ):
        if limit is not None and (
            isinstance(limit, bool)
            or not isinstance(limit, (int, float))
            or not math.isfinite(limit)
            or limit < 0
        ):
            raise ValueError(f"{name} must be a finite nonnegative number")
    if backend == "static":
        if inner_time_limit is not None:
            raise ValueError("inner_time_limit is supported only by master_oracle")
        if precompute_invariants or task_local_cache:
            raise ValueError("options are supported only by master_oracle")
        return solve_static(
            instance,
            time_limit=time_limit,
            total_time_limit=total_time_limit,
        )
    from .solvers import MasterOracleOptions

    return solve_master_oracle(
        instance,
        time_limit=time_limit,
        inner_time_limit=inner_time_limit,
        total_time_limit=total_time_limit,
        options=MasterOracleOptions(
            precompute_invariants=precompute_invariants,
            task_local_cache=task_local_cache,
        ),
    )


def _package_version() -> str:
    try:
        return version("pess-ess")
    except PackageNotFoundError:
        return "unknown"


def _solver_status(item: Any) -> dict[str, Any]:
    return {
        "status_code": item.status,
        "message": item.message,
        "solver_reported_optimal": item.optimal,
        "incumbent_returned": item.incumbent_returned,
        "primal_objective": item.primal_objective,
        "dual_bound": item.dual_bound,
        "mip_gap": item.mip_gap,
        "duration_seconds": item.total_duration_seconds,
        "highs_invocations": item.highs_invocations,
        "selected_attempt": item.selected_attempt,
    }


def result_to_dict(
    result: OuterResult,
    *,
    backend: Backend,
    objective: str = "pess",
) -> dict[str, Any]:
    """Serialize validated witnesses, numerical status, and provenance."""
    selected = sorted(result.selected) if result.selected is not None else None
    selected_pess_value_certified = bool(
        result.validated_feasible
        and result.objective is not None
        and (
            backend == "static" or result.exact_oracle_invocations == 0
        )
        and not any(
            "task maximum is not Fraction-certified" in warning
            for warning in result.numerical_warnings
        )
    )
    bound_scope = (
        "complete_static_formulation"
        if backend == "static"
        else "restricted_master_at_termination"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "objective": objective,
        "backend": backend,
        "status": result.status,
        "termination_reason": result.termination_reason,
        "feasibility": (
            "feasible" if result.validated_feasible else "not_validated"
        ),
        "optimality": {
            "solver_reported_optimal": result.solver_reported_optimal,
            "kind": "floating_point_milp_status",
            "solver_reported_lower_bound": result.lower_bound,
            "lower_bound_scope": bound_scope,
            "lower_bound_is_mathematical_certificate": False,
            "global_incumbent_upper_bound": (
                str(result.objective) if selected_pess_value_certified else None
            ),
            "global_upper_bound_is_fraction_certified": selected_pess_value_certified,
            "mip_gap": (
                result.master_mip_gap
                if result.status == "optimal" and result.solver_reported_optimal
                and selected_pess_value_certified
                and not result.numerical_warnings
                else None
            ),
            "outer_global_optimum_fraction_certified": False,
        },
        "incumbent": {
            "available": selected is not None,
            "validated_feasible": result.validated_feasible,
            "selected": selected,
            "maintenance_cost": (
                str(result.maintenance_cost)
                if result.maintenance_cost is not None
                else None
            ),
            "witness_aggregate": (
                str(result.objective) if result.objective is not None else None
            ),
        },
        "objective_evidence": {
            "witness_losses_aggregated_with_fraction_arithmetic": True,
            "selected_pess_value_fraction_certified": selected_pess_value_certified,
            "basis": (
                "complete_fraction_matching_enumeration"
                if backend == "static"
                else (
                    "fraction_certified_inner_bounds"
                    if result.exact_oracle_invocations == 0
                    else "witnesses_with_numerical_inner_mip_status"
                )
            ),
            "numerical_inner_oracle_invocations_during_run": (
                result.exact_oracle_invocations
            ),
            "inner_mip_solve_calls": result.inner_mip_solve_calls,
        },
        "solver_statuses": [_solver_status(item) for item in result.solver_statuses],
        "wall_time_seconds": result.total_runtime_seconds,
        "provenance": {
            "package": "pess-ess",
            "package_version": _package_version(),
            "milp_api": "scipy.optimize.milp",
            "scipy_version": scipy.__version__,
            "backend_solver": "HiGHS via SciPy",
            "highs_version": None,
            "highs_version_note": "SciPy milp does not expose its bundled HiGHS version",
        },
        "diagnostics": {
            "master_iterations": result.master_iterations,
            "distinct_cuts": result.distinct_cuts,
            "scenario_count": result.scenario_count,
            "numerical_warnings": list(result.numerical_warnings),
        },
    }


def solve_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a versioned JSON request and return its versioned result."""
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
    objective = payload.get("objective", "pess")
    backend = payload.get("backend", "master_oracle")
    limits = payload.get("limits", {})
    options = payload.get("options", {})
    provenance = payload.get("input_provenance", {})
    if not isinstance(limits, dict) or not isinstance(options, dict):
        raise TypeError("limits and options must be JSON objects")
    if not isinstance(provenance, dict):
        raise TypeError("input_provenance must be a JSON object")
    supported_limits = {
        "time_seconds",
        "inner_time_seconds",
        "total_time_seconds",
    }
    supported_options = {"precompute_invariants", "task_local_cache"}
    if set(limits) - supported_limits:
        raise ValueError(f"unsupported limit keys: {sorted(set(limits) - supported_limits)}")
    if set(options) - supported_options:
        raise ValueError(
            f"unsupported option keys: {sorted(set(options) - supported_options)}"
        )
    if any(not isinstance(value, bool) for value in options.values()):
        raise TypeError("all options must be booleans")
    instance = instance_from_dict(payload["instance"])
    result = solve(
        instance,
        objective=objective,
        backend=backend,
        time_limit=limits.get("time_seconds"),
        inner_time_limit=limits.get("inner_time_seconds"),
        total_time_limit=limits.get("total_time_seconds"),
        precompute_invariants=options.get("precompute_invariants", False),
        task_local_cache=options.get("task_local_cache", False),
    )
    response = result_to_dict(result, objective=objective, backend=backend)
    response["input_provenance"] = provenance
    return response
