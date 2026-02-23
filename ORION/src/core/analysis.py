"""Beam analysis and caustic fitting."""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np

from ORION.config import Config
from ORION.src.core.types import BeamAnalysis, EMPTY_BEAM

logger = logging.getLogger(__name__)


class BeamAnalyzer:
    """Analyze beam size and orientation using ISO 11146 second moments."""

    def __init__(self, config: Config):
        self.config = config

    @staticmethod
    def _weighted_moments(weights: np.ndarray) -> tuple[float, float, float, float, float]:
        """Return total, centroid (x/y), and second moments (u_xx, u_yy) for a weighted 2D map."""
        h, w = weights.shape
        row_sums = weights.sum(axis=1, dtype=np.float64)
        col_sums = weights.sum(axis=0, dtype=np.float64)
        total = float(row_sums.sum(dtype=np.float64))
        if total <= 0.0:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        x_idx = np.arange(w, dtype=np.float64)
        y_idx = np.arange(h, dtype=np.float64)

        cx = float(np.dot(col_sums, x_idx) / total)
        cy = float(np.dot(row_sums, y_idx) / total)
        u_xx = float(np.dot(col_sums, (x_idx - cx) ** 2) / total)
        u_yy = float(np.dot(row_sums, (y_idx - cy) ** 2) / total)
        return total, cx, cy, u_xx, u_yy

    @staticmethod
    def _mad_sigma(values: np.ndarray) -> float:
        return 1.4826 * float(np.median(np.abs(values)))

    def analyze_beam(self, img: np.ndarray, max_val: float, virtual_pixel_size: float) -> BeamAnalysis:
        """
        Estimate beam size and centroid.

        Returns:
            BeamAnalysis(d4s_eff_um, d4s_x_um, d4s_y_um, azimuth_deg,
                         centroid_x_px, centroid_y_px, d4s_x_px, d4s_y_px)
        """
        try:
            full_h, full_w = img.shape

            stride = 2
            data_coarse = img[::stride, ::stride].astype(np.float32, copy=False)
            baseline = float(np.median(data_coarse))
            data_coarse -= baseline

            sigma = self._mad_sigma(data_coarse)
            peak = float(np.max(data_coarse))
            min_signal = max(float(self.config.LOW_SIGNAL_THRESHOLD), 6.0 * sigma)
            low_signal_mode = peak < min_signal

            cutoff = max(5.0, peak * self.config.NOISE_CUTOFF_PERCENT, 6.0 * sigma)
            mask_coarse = data_coarse > cutoff
            if mask_coarse.size:
                mask_frac = float(np.count_nonzero(mask_coarse)) / float(mask_coarse.size)
                if mask_frac > 0.5:
                    cutoff = max(cutoff, peak * 0.7)
                    mask_coarse = data_coarse > cutoff

            found_coarse = np.count_nonzero(mask_coarse) >= 5

            if found_coarse and not low_signal_mode:
                weights = np.where(mask_coarse, data_coarse, 0.0)
                total, cx_s, cy_s, u_xx_s, u_yy_s = self._weighted_moments(weights)
                if total <= 0.0:
                    return EMPTY_BEAM

                cnt_x = cx_s * stride
                cnt_y = cy_s * stride

                dev_x = np.sqrt(u_xx_s) * stride * 4.0
                dev_y = np.sqrt(u_yy_s) * stride * 4.0
            else:
                if low_signal_mode:
                    by, bx = np.unravel_index(np.argmax(data_coarse), data_coarse.shape)
                    cnt_x = float(bx * stride)
                    cnt_y = float(by * stride)

                    win = max(25, min(75, min(full_w, full_h) // 8))
                    half = win // 2
                    x0 = max(0, int(cnt_x) - half)
                    y0 = max(0, int(cnt_y) - half)
                    x1 = min(full_w, x0 + win)
                    y1 = min(full_h, y0 + win)
                    x0 = max(0, x1 - win)
                    y0 = max(0, y1 - win)

                    sub = img[y0:y1, x0:x1].astype(np.float32, copy=False)
                    sub -= float(np.median(sub))
                    sigma_sub = self._mad_sigma(sub)
                    thr = max(1.0, 3.0 * sigma_sub)
                    sub[sub < thr] = 0.0

                    total, cx_local, cy_local, u_xx, u_yy = self._weighted_moments(sub)
                    if total <= 0.0:
                        return EMPTY_BEAM

                    cnt_x = x0 + cx_local
                    cnt_y = y0 + cy_local
                    dev_x = max(np.sqrt(u_xx), 1.0)
                    dev_y = max(np.sqrt(u_yy), 1.0)
                else:
                    data_full = img.astype(np.float32, copy=False)
                    data_full -= float(np.median(data_full))

                    sigma_full = self._mad_sigma(data_full)
                    peak_full = float(np.max(data_full))
                    min_signal_full = max(float(self.config.LOW_SIGNAL_THRESHOLD), 6.0 * sigma_full)
                    if peak_full < min_signal_full:
                        return EMPTY_BEAM

                    cutoff_full = max(5.0, peak_full * self.config.NOISE_CUTOFF_PERCENT, 6.0 * sigma_full)
                    mask_full = data_full > cutoff_full
                    if np.count_nonzero(mask_full) < 5:
                        return EMPTY_BEAM

                    y_vals, x_vals = np.nonzero(mask_full)
                    pixel_values = data_full[y_vals, x_vals]
                    pv64 = pixel_values.astype(np.float64, copy=False)
                    total_f = float(np.sum(pv64, dtype=np.float64))

                    if total_f == 0.0:
                        return EMPTY_BEAM

                    cnt_x = float(np.dot(x_vals.astype(np.float64), pv64) / total_f)
                    cnt_y = float(np.dot(y_vals.astype(np.float64), pv64) / total_f)
                    dev_x, dev_y = 10.0, 10.0

            d4s_x_px = 0.0
            d4s_y_px = 0.0
            azimuth_deg = 0.0

            for _ in range(2):
                margin_factor = 12.0
                wx_roi = max(50.0, margin_factor * dev_x)
                wy_roi = max(50.0, margin_factor * dev_y)

                min_x = max(0, int(cnt_x - wx_roi))
                max_x = min(full_w, int(cnt_x + wx_roi))
                min_y = max(0, int(cnt_y - wy_roi))
                max_y = min(full_h, int(cnt_y + wy_roi))

                if (max_x - min_x) < 5 or (max_y - min_y) < 5:
                    break

                sub_data = img[min_y:max_y, min_x:max_x].astype(np.float32, copy=False)
                sub_data -= baseline

                integration_cutoff = max(1.0, max_val * 0.002)
                sub_data[sub_data <= integration_cutoff] = 0.0

                total, cx_local, cy_local, u_xx, u_yy = self._weighted_moments(sub_data)
                if total <= 0.0:
                    break

                cnt_x = min_x + cx_local
                cnt_y = min_y + cy_local

                sub_h, sub_w = sub_data.shape
                x_idx = np.arange(sub_w, dtype=np.float32)
                y_idx = np.arange(sub_h, dtype=np.float32)
                sum_xy = float(y_idx @ sub_data @ x_idx)
                u_xy = (sum_xy / total) - (cx_local * cy_local)

                dev_x = np.sqrt(u_xx)
                dev_y = np.sqrt(u_yy)

                diff = u_xx - u_yy
                sum_var = u_xx + u_yy

                phi = 0.5 * np.arctan2(2 * u_xy, diff)
                azimuth_deg = float(np.degrees(phi))

                term = np.sqrt(diff**2 + 4 * u_xy**2)
                d1 = 2 * np.sqrt(2) * np.sqrt(max(0.0, sum_var + term))
                d2 = 2 * np.sqrt(2) * np.sqrt(max(0.0, sum_var - term))

                if abs(azimuth_deg) <= 45.0:
                    d4s_x_px, d4s_y_px = d1, d2
                else:
                    d4s_x_px, d4s_y_px = d2, d1

            d4s_eff_px = float(np.sqrt(d4s_x_px * d4s_y_px))

            max_reasonable = 0.9 * float(min(full_w, full_h))
            if d4s_x_px > max_reasonable or d4s_y_px > max_reasonable:
                return EMPTY_BEAM

            pixel_um = float(virtual_pixel_size)
            return BeamAnalysis(
                d4s_eff_px * pixel_um,
                d4s_x_px * pixel_um,
                d4s_y_px * pixel_um,
                azimuth_deg,
                float(cnt_x),
                float(cnt_y),
                float(d4s_x_px),
                float(d4s_y_px),
            )
        except Exception:
            logger.exception("Beam analysis failed")
            return EMPTY_BEAM

    def calculate_caustic_fit(self, z_values: Iterable[float], d_values: Iterable[float], wavelength_nm: float) -> dict:
        """Fit a hyperbola to caustic data to estimate M^2 parameters."""
        try:
            z = np.array(list(z_values), dtype=np.float64)
            d = np.array(list(d_values), dtype=np.float64)

            if len(z) < 3:
                return {}

            y = d**2
            coeffs = np.polyfit(z, y, 2)
            C, B, A = coeffs[0], coeffs[1], coeffs[2]
            if C <= 0:
                return {"error": "Invalid fit (negative curvature)"}

            z0 = -B / (2 * C)
            d0_sq = A - C * (z0**2)
            d0 = float(np.sqrt(d0_sq)) if d0_sq > 0 else 0.0

            theta_rad = float(np.sqrt(C))
            theta_mrad = theta_rad * 1000.0
            zR = d0 / theta_rad if theta_rad > 0 else 0.0

            lam_mm = wavelength_nm * 1.0e-6
            m2 = (np.pi * d0 * theta_rad) / (4 * lam_mm)

            z_min = min(np.min(z), z0 - 2 * zR)
            z_max = max(np.max(z), z0 + 2 * zR)
            range_z = z_max - z_min or 1.0
            fit_z = np.linspace(z_min - range_z * 0.1, z_max + range_z * 0.1, 100)

            fit_d_sq = d0**2 + (theta_rad**2) * (fit_z - z0) ** 2
            fit_d = np.sqrt(fit_d_sq)

            return {
                "success": True,
                "M2": m2,
                "d0_mm": d0,
                "z0_mm": z0,
                "zR_mm": zR,
                "theta_mrad": theta_mrad,
                "d0": d0 * 1000.0,
                "z0": z0,
                "div": theta_mrad,
                "zr": zR,
                "fit_z": fit_z,
                "fit_d": fit_d,
                "raw_z": z,
                "raw_d": None,
            }
        except Exception:
            logger.exception("Caustic fit failed")
            return {"error": "Fit failed"}
