"""Shared core data structures used across analysis and orchestration."""

from __future__ import annotations

from typing import NamedTuple


class BeamAnalysis(NamedTuple):
    """Beam measurement in physical and pixel units."""

    d4s_eff_um: float
    d4s_x_um: float
    d4s_y_um: float
    azimuth_deg: float
    centroid_x_px: float
    centroid_y_px: float
    d4s_x_px: float
    d4s_y_px: float


EMPTY_BEAM = BeamAnalysis(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
