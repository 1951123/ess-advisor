"""Public outer-solver API.

The implementation is kept in solvers_v2 to preserve the historical import
path while making diagnostics easy to audit.
"""

from .solvers_v2 import (
    MasterOracleOptions,
    OuterResult,
    TaskLocalOracleCache,
    solve_master_oracle,
    solve_static,
)

__all__ = [
    "MasterOracleOptions",
    "OuterResult",
    "TaskLocalOracleCache",
    "solve_master_oracle",
    "solve_static",
]
