"""Shared constants for stage-informed imputations and defaults."""

from __future__ import annotations

T_STAGE_PRIOR_SIZE = {
    "T1": 2.0,
    "T2": 3.5,
    "T3": 5.0,
    "T4": 6.5,
}

N_STAGE_PRIOR_LN_RATIO = {
    "N0": 0.02,
    "N1": 0.15,
    "N2": 0.35,
    "N3": 0.65,
}

DEFAULT_T_STAGE = "T2"
DEFAULT_N_STAGE = "N1"
