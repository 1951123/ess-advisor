"""Callable and JSON contract for supported structured ESS objectives."""

from __future__ import annotations

import math
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Literal

import scipy

from .model import Instance, instance_from_dict
from .nominal import solve_nominal
from .solvers import OuterResult, solve_master_oracle, solve_static

Backend = Literal["master_oracle", "static", "fixed_priority_milp"]
SCHEMA_VERSION = "1"


def solve(
    instance: Instance,
    *,
    objective: str = "pess",
    backend: Backend | None = None,
    priority_order: list[str] | tuple[str, ...] | None = None,
    time_limit: float | None = None,
    inner_time_limit: float | None = None,
    total_time_limit: float | None = None,
    precompute_invariants: bool = False,
    task_local_cache: bool = False,
) -> OuterResult:
    """Solve a supported structured ESS objective using conventional MILPs.

    PESS defaults to Master--Oracle, which does not enumerate outer resource
    sets. Nominal uses a complete fixed candidate-priority order shared by all
    tasks and a direct deterministic-policy MILP.
    """
    if objective not in {"pess", "nominal"}:
        raise ValueError("objective must be 'pess' or 'nominal'")
    if backend is None:
        backend = "fixed_priority_milp" if objective == "nominal" else "master_oracle"
    if objective == "pess" and backend not in {"master_oracle", "static"}:
        raise ValueError("PESS backend must be 'master_oracle' or 'static'")
    if objective == "nominal" and backend != "fixed_priority_milp":
        raise ValueError("Nominal backend must be 'fixed_priority_milp'")
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
    if objective == "nominal":
        if priority_order is None:
            raise ValueError("objective='nominal' requires priority_order")
        if inner_time_limit is not None or precompute_invariants or task_local_cache:
            raise ValueError(
                "inner_time_limit and master-oracle options apply only to PESS"
            )
        return solve_nominal(
            instance,
            priority_order,
            time_limit=time_limit,
            total_time_limit=total_time_limit,
        )
    if priority_order is not None:
        raise ValueError("priority_order is supported only by objective='nominal'")
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
    priority_order: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Serialize validated objective evidence, numerical status, and provenance."""
    if objective not in {"pess", "nominal"}:
        raise ValueError("objective must be 'pess' or 'nominal'")
    selected = sorted(result.selected) if result.selected is not None else None
    selected_nominal_value_certified = bool(
        objective == "nominal"
        and backend == "fixed_priority_milp"
        and result.validated_feasible
        and result.objective is not None
    )
    selected_pess_value_certified = bool(
        objective == "pess"
        and result.validated_feasible
        and result.objective is not None
        and (backend == "static" or result.exact_oracle_invocations == 0)
        and not any(
            "task maximum is not Fraction-certified" in warning
            for warning in result.numerical_warnings
        )
    )
    selected_value_certified = (
        selected_nominal_value_certified or selected_pess_value_certified
    )
    bound_scope = (
        "complete_static_formulation"
        if backend == "static"
        else (
            "complete_fixed_priority_formulation"
            if backend == "fixed_priority_milp"
            else "restricted_master_at_termination"
        )
    )
    if objective == "nominal":
        result_priority = result.policy_priority_order
        if result_priority is None:
            raise ValueError("Nominal result is missing its policy priority order")
        if priority_order is not None and tuple(priority_order) != result_priority:
            raise ValueError("priority_order does not match the solved Nominal policy")
        priority_order = result_priority
        objective_evidence = {
            "selected_nominal_value_fraction_certified": selected_nominal_value_certified,
            "basis": "fixed_global_priority_greedy_replay",
            "task_losses_aggregated_with_fraction_arithmetic": True,
            "priority_order_shared_across_tasks": True,
        }
        incumbent_objective_key = "objective_value"
    else:
        objective_evidence = {
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
        }
        incumbent_objective_key = "witness_aggregate"
    incumbent = {
        "available": selected is not None,
        "validated_feasible": result.validated_feasible,
        "selected": selected,
        "maintenance_cost": (
            str(result.maintenance_cost)
            if result.maintenance_cost is not None
            else None
        ),
        incumbent_objective_key: (
            str(result.objective) if result.objective is not None else None
        ),
    }
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
                str(result.objective) if selected_value_certified else None
            ),
            "global_upper_bound_is_fraction_certified": selected_value_certified,
            "mip_gap": (
                result.master_mip_gap
                if result.status == "optimal" and result.solver_reported_optimal
                and selected_value_certified
                and not result.numerical_warnings
                else None
            ),
            "outer_global_optimum_fraction_certified": False,
        },
        "incumbent": incumbent,
        "objective_evidence": objective_evidence,
        **(
            {
                "policy": {
                    "kind": "fixed_global_candidate_priority_greedy",
                    "priority_order": list(priority_order or ()),
                }
            }
            if objective == "nominal"
            else {}
        ),
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
    backend = payload.get("backend")
    priority_order = payload.get("priority_order")
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
        priority_order=priority_order,
        time_limit=limits.get("time_seconds"),
        inner_time_limit=limits.get("inner_time_seconds"),
        total_time_limit=limits.get("total_time_seconds"),
        precompute_invariants=options.get("precompute_invariants", False),
        task_local_cache=options.get("task_local_cache", False),
    )
    resolved_backend: Backend = backend or (
        "fixed_priority_milp" if objective == "nominal" else "master_oracle"
    )
    response = result_to_dict(
        result,
        objective=objective,
        backend=resolved_backend,
        priority_order=priority_order,
    )
    response["input_provenance"] = provenance
    return response
