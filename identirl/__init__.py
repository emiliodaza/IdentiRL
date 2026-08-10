"""IdentiRL: controlled causal diagnosis benchmarks for reinforcement learning."""

from .envs import Condition, DiagnosticPOMDP, PassiveAmbiguityEnv
from .theory import ValueGaps, factorial_value_gaps, matched_value_gaps

__all__ = [
    "Condition",
    "DiagnosticPOMDP",
    "PassiveAmbiguityEnv",
    "ValueGaps",
    "factorial_value_gaps",
    "matched_value_gaps",
]

__version__ = "0.1.0"
