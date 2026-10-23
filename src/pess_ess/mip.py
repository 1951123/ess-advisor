"""Small SciPy/HiGHS mixed-integer-programming adapter with diagnostics."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, csr_matrix


@dataclass(frozen=True)
class MIPAttempt:
    status: int
    message: str
    primal_objective: float | None
    dual_bound: float | None
    mip_gap: float | None
    mip_node_count: int | None
    values: tuple[float, ...]
    duration_seconds: float
    invocation_performed: bool
    presolve: bool | None
    incumbent_returned: bool
    optimality_reported: bool


@dataclass(frozen=True)
class MIPResult:
    status: int
    message: str
    primal_objective: float | None
    dual_bound: float | None
    values: tuple[float, ...]
    mip_gap: float | None
    mip_node_count: int | None
    initial_attempt: MIPAttempt
    retry_attempt: MIPAttempt | None
    selected_attempt: str
    total_duration_seconds: float
    highs_invocations: int
    presolve_fallback_attempted: bool
    presolve_fallback_successes: int

    @property
    def objective(self) -> float | None:
        """Backward-compatible alias for the primal incumbent objective."""
        return self.primal_objective

    @property
    def optimal(self) -> bool:
        return (
            self.status == 0 and self.attempt(self.selected_attempt).optimality_reported
        )

    @property
    def incumbent_returned(self) -> bool:
        return bool(self.values) or self.primal_objective is not None

    def attempt(self, name: str) -> MIPAttempt:
        if name == "retry" and self.retry_attempt is not None:
            return self.retry_attempt
        return self.initial_attempt


SolverRunner = Callable[..., Any]


def _sparse_constraint_matrix(
    rows: list[tuple[dict[int, float], float, float]],
    column_count: int,
) -> csr_matrix:
    """Build the row-dictionary constraints without a dense intermediate."""
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    for row_index, (coefficients, _lower, _upper) in enumerate(rows):
        for variable_index, coefficient in coefficients.items():
            # Match NumPy's former dense assignment behavior for negative indices.
            normalized_index = variable_index
            if normalized_index < 0:
                normalized_index += column_count
            if normalized_index < 0 or normalized_index >= column_count:
                raise IndexError("constraint variable index out of range")
            row_indices.append(row_index)
            column_indices.append(normalized_index)
            values.append(coefficient)
    return coo_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(rows), column_count),
        dtype=float,
    ).tocsr()


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if np.isfinite(converted) else None


def _attempt_from_result(
    result: Any,
    duration_seconds: float,
    presolve: bool | None,
) -> MIPAttempt:
    raw_values = getattr(result, "x", None)
    values = (
        tuple(float(value) for value in raw_values) if raw_values is not None else ()
    )
    status = int(getattr(result, "status", 4))
    primal = _finite_float(getattr(result, "fun", None))
    node_count = getattr(result, "mip_node_count", None)
    return MIPAttempt(
        status=status,
        message=str(getattr(result, "message", "unknown HiGHS result")),
        primal_objective=primal,
        dual_bound=_finite_float(getattr(result, "mip_dual_bound", None)),
        mip_gap=_finite_float(getattr(result, "mip_gap", None)),
        mip_node_count=int(node_count) if node_count is not None else None,
        values=values,
        duration_seconds=duration_seconds,
        invocation_performed=True,
        presolve=presolve,
        incumbent_returned=raw_values is not None and primal is not None,
        optimality_reported=status == 0,
    )


def _timeout_attempt(message: str, presolve: bool | None) -> MIPAttempt:
    return MIPAttempt(
        status=1,
        message=message,
        primal_objective=None,
        dual_bound=None,
        mip_gap=None,
        mip_node_count=None,
        values=(),
        duration_seconds=0.0,
        invocation_performed=False,
        presolve=presolve,
        incumbent_returned=False,
        optimality_reported=False,
    )


def solve_binary_milp(
    objective: Iterable[float],
    integrality: Iterable[int],
    lower_bounds: Iterable[float],
    upper_bounds: Iterable[float],
    rows: list[tuple[dict[int, float], float, float]],
    time_limit: float | None = None,
    *,
    deadline: float | None = None,
    clock: Callable[[], float] = time.perf_counter,
    solver_runner: SolverRunner | None = None,
) -> MIPResult:
    """Solve one MILP while preserving primal/dual and retry diagnostics.

    time_limit is a per-call limit. deadline is an absolute monotonic
    deadline shared by an outer algorithm and a presolve retry.
    """
    objective_array = np.asarray(tuple(objective), dtype=float)
    integral_array = np.asarray(tuple(integrality), dtype=int)
    lower_array = np.asarray(tuple(lower_bounds), dtype=float)
    upper_array = np.asarray(tuple(upper_bounds), dtype=float)
    if len(objective_array) != len(integral_array):
        raise ValueError("objective and integrality lengths differ")
    if len(lower_array) != len(objective_array):
        raise ValueError("variable bound lengths differ")
    matrix = _sparse_constraint_matrix(rows, len(objective_array))
    row_lower = np.full(len(rows), -np.inf, dtype=float)
    row_upper = np.full(len(rows), np.inf, dtype=float)
    for row_index, (coefficients, lower, upper) in enumerate(rows):
        for variable_index, coefficient in coefficients.items():
            matrix[row_index, variable_index] = coefficient
        row_lower[row_index] = lower
        row_upper[row_index] = upper
    constraints = LinearConstraint(matrix, row_lower, row_upper) if rows else None
    bounds = Bounds(lower_array, upper_array)
    runner = solver_runner or milp
    started = clock()
    operation_deadline = deadline
    if time_limit is not None:
        local_deadline = started + max(0.0, float(time_limit))
        operation_deadline = (
            local_deadline
            if operation_deadline is None
            else min(operation_deadline, local_deadline)
        )

    def remaining() -> float | None:
        if operation_deadline is None:
            return None
        return max(0.0, operation_deadline - clock())

    def run_once(presolve: bool | None) -> MIPAttempt:
        available = remaining()
        if available is not None and available <= 0.0:
            return _timeout_attempt(
                "deadline exhausted before HiGHS invocation",
                presolve,
            )
        run_options: dict[str, object] = {}
        if available is not None:
            run_options["time_limit"] = available
        if presolve is False:
            run_options["presolve"] = False
        call_started = clock()
        raw_result = runner(
            c=objective_array,
            integrality=integral_array,
            bounds=bounds,
            constraints=constraints,
            options=run_options,
        )
        return _attempt_from_result(
            raw_result,
            clock() - call_started,
            presolve,
        )

    initial = run_once(None)
    retry = None
    if initial.status == 4 and (remaining() is None or remaining() > 0.0):
        retry = run_once(False)
    if retry is not None:
        if retry.incumbent_returned or retry.optimality_reported:
            selected_name, selected = "retry", retry
        elif initial.incumbent_returned:
            selected_name, selected = "initial", initial
        else:
            selected_name, selected = "retry", retry
    else:
        selected_name, selected = "initial", initial
    message = selected.message
    if retry is not None:
        message = (
            f"initial status {initial.status}: {initial.message}; "
            f"retry status {retry.status}: {retry.message}"
        )
    return MIPResult(
        status=selected.status,
        message=message,
        primal_objective=selected.primal_objective,
        dual_bound=selected.dual_bound,
        values=selected.values,
        mip_gap=selected.mip_gap,
        mip_node_count=selected.mip_node_count,
        initial_attempt=initial,
        retry_attempt=retry,
        selected_attempt=selected_name,
        total_duration_seconds=clock() - started,
        highs_invocations=int(initial.invocation_performed)
        + (int(retry.invocation_performed) if retry is not None else 0),
        presolve_fallback_attempted=retry is not None,
        presolve_fallback_successes=int(retry is not None and retry.status == 0),
    )
