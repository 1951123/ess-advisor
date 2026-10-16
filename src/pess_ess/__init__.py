"""Synthetic PESS-ESS mathematical model and exact small-instance solvers."""

from .model import Candidate, Instance, Task, read_instance, write_instance
from .oracle import (
    InnerBounds,
    InnerOracleResult,
    evaluate_task,
    ordinary_matching_extrema,
)
from .reference import ReferenceResult, solve_exhaustive
from .solvers import OuterResult, solve_master_oracle, solve_static

__all__ = [
    "Candidate",
    "InnerBounds",
    "InnerOracleResult",
    "Instance",
    "OuterResult",
    "ReferenceResult",
    "Task",
    "evaluate_task",
    "ordinary_matching_extrema",
    "read_instance",
    "solve_exhaustive",
    "solve_master_oracle",
    "solve_static",
    "write_instance",
]
