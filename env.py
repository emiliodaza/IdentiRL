"""Backward-compatible environment imports."""

from identirl.envs import Condition, DiagnosticPOMDP, PassiveAmbiguityEnv

TinyDiagnosticPOMDP = DiagnosticPOMDP

__all__ = ["Condition", "DiagnosticPOMDP", "PassiveAmbiguityEnv", "TinyDiagnosticPOMDP"]
