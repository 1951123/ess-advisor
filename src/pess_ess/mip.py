"""Small SciPy/HiGHS mixed-integer linear-programming adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


@dataclass(frozen=True)
class MIPResult:
    status: int
    message: str
    objective: float | None
    values: tuple[float, ...]
    mip_gap: float | None
    mip_node_count: int | None

    @property
    def optimal(self) -> bool:
        return self.status == 0


def solve_binary_milp(
    objective: Iterable[float],
    integrality: Iterable[int],
    lower_bounds: Iterable[float],
    upper_bounds: Iterable[float],
    rows: list[tuple[dict[int, float], float, float]],
    time_limit: float | None = None,
) -> MIPResult:
    objective_array = np.asarray(tuple(objective), dtype=float)
    integral_array = np.asarray(tuple(integrality), dtype=int)
    lower_array = np.asarray(tuple(lower_bounds), dtype=float)
    upper_array = np.asarray(tuple(upper_bounds), dtype=float)
    if len(objective_array) != len(integral_array):
        raise ValueError("objective and integrality lengths differ")
    if len(lower_array) != len(objective_array):
        raise ValueError("variable bound lengths differ")
    matrix = np.zeros((len(rows), len(objective_array)), dtype=float)
    row_lower = np.full(len(rows), -np.inf, dtype=float)
    row_upper = np.full(len(rows), np.inf, dtype=float)
    for row_index, (coefficients, lower, upper) in enumerate(rows):
        for variable_index, coefficient in coefficients.items():
            matrix[row_index, variable_index] = coefficient
        row_lower[row_index] = lower
        row_upper[row_index] = upper
    constraints = (
        LinearConstraint(matrix, row_lower, row_upper)
        if rows
        else None
    )
    options = {}
    if time_limit is not None:
        options["time_limit"] = float(time_limit)
    def run_milp(run_options: dict[str, object]):
        return milp(
            c=objective_array,
            integrality=integral_array,
            bounds=Bounds(lower_array, upper_array),
            constraints=constraints,
            options=run_options,
        )

    result = run_milp(options)
    if result.status == 4:
        fallback_options = dict(options)
        fallback_options["presolve"] = False
        result = run_milp(fallback_options)
    values = tuple(float(value) for value in result.x) if result.x is not None else ()
    raw_gap = getattr(result, "mip_gap", None)
    mip_gap = (
        float(raw_gap)
        if raw_gap is not None and np.isfinite(float(raw_gap))
        else None
    )
    return MIPResult(
        status=int(result.status),
        message=str(result.message),
        objective=float(result.fun) if result.fun is not None else None,
        values=values,
        mip_gap=mip_gap,
        mip_node_count=int(result.mip_node_count)
        if result.mip_node_count is not None
        else None,
    )
