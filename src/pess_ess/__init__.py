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
    "Instance",
    "Task",
    "read_instance",
    "write_instance",
    "InnerBounds",
    "InnerOracleResult",
    "evaluate_task",
    "ordinary_matching_extrema",
    "ReferenceResult",
    "solve_exhaustive",
    "OuterResult",
    "solve_master_oracle",
    "solve_static",
]
