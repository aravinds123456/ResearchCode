"""Branching Hallucinations experiment package.

Ground-truth construction, tree generation, trajectory measurement, and
analysis are separate stages with separate artifacts.
"""

from .schemas import (
    SCHEMA_VERSION,
    Action,
    TrajectoryState,
    VerificationStatus,
)

__all__ = [
    "SCHEMA_VERSION",
    "Action",
    "TrajectoryState",
    "VerificationStatus",
]
