import numpy as np
import traceback
from ORION.config import Config

class BeamAnalyzer:
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

    def analyze_beam(self, img: np.ndarray, max_val: float, virtual_pixel_size: float) -> tuple:
        """
        Calculates beam widths using ISO 11146 2nd moment method.
        Optimized with strided search and ROI slicing.
        Returns: (d4s_eff, d4s_x, d4s_y, azimuth_deg, cnt_x, cnt_y, d4s_x_px, d4s_y_px)
        """
        try:
            full_h, full_w = img.shape
            
            # 1. Initial Coarse Search (Strided)
            # Use stride of 2 (instead of 4) to prevent aliasing small beams (focus spots).
            # Stride 4 was missing beams < ~5px spread.
            stride = 2
            data_coarse = img[::stride, ::stride].astype(np.float32, copy=False)
            baseline = float(np.median(data_coarse))
            data_coarse -= baseline

            # Robust noise estimate (MAD) on baseline-subtracted data
            mad = float(np.median(np.abs(data_coarse)))
            sigma = 1.4826 * mad
            peak = float(np.max(data_coarse))

            # Low-signal mode: use local-window analysis instead of global mask
            min_signal = max(float(self.config.LOW_SIGNAL_THRESHOLD), 6.0 * sigma)
            low_signal_mode = peak < min_signal

            cutoff = max(5.0, peak * self.config.NOISE_CUTOFF_PERCENT, 6.0 * sigma)
            mask_coarse = data_coarse > cutoff

            # If threshold is too low (mask dominates), tighten it once.
            if mask_coarse.size > 0:
                mask_frac = float(np.count_nonzero(mask_coarse)) / float(mask_coarse.size)
                if mask_frac > 0.5:
                    cutoff = max(cutoff, peak * 0.7)
                    mask_coarse = data_coarse > cutoff
            
            found_coarse = np.count_nonzero(mask_coarse) >= 5
            
            if found_coarse and not low_signal_mode:
                weights_s = np.where(mask_coarse, data_coarse, 0.0)
                total_s, cx_s, cy_s, u_xx_s, u_yy_s = self._weighted_moments(weights_s)
                if total_s <= 0.0:
                    return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                
                cnt_x = cx_s * stride
                cnt_y = cy_s * stride

                # Note: This sigma is calculated ONLY on pixels > cutoff.
                # Real Gaussian wings extend much further. 
                # This underestimates sigma by ~30-50% depending on cutoff.
                
                # Apply 4.0x safety factor because the mask cuts off wings drastically
                dev_x = np.sqrt(u_xx_s) * stride * 4.0
                dev_y = np.sqrt(u_yy_s) * stride * 4.0
                
            else:
                # FALLBACK: Small Beam Detection
                # If the beam is tiny (e.g. < 4px), stride=4 might skip it.
                # Check full image, but use sparse calculation to remain fast.
                if low_signal_mode:
                    # Local-window low-signal analysis around peak
                    by, bx = np.unravel_index(np.argmax(data_coarse), data_coarse.shape)
                    cnt_x = float(bx * stride)
                    cnt_y = float(by * stride)

                    # Window size tuned for small beams; clamp to image
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
                    mad_sub = float(np.median(np.abs(sub)))
                    sigma_sub = 1.4826 * mad_sub
                    thr = max(1.0, 3.0 * sigma_sub)
                    sub[sub < thr] = 0.0

                    total_w, cx_local, cy_local, u_xx, u_yy = self._weighted_moments(sub)
                    if total_w <= 0.0:
                        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

                    cnt_x = x0 + cx_local
                    cnt_y = y0 + cy_local
                    dev_x = max(np.sqrt(u_xx), 1.0)
                    dev_y = max(np.sqrt(u_yy), 1.0)
                else:
                    data_full = img.astype(np.float32, copy=False)
                    # Recalculate baseline and noise on full img to be safe
                    base_full = float(np.median(data_full))
                    data_full -= base_full

                    mad_full = float(np.median(np.abs(data_full)))
                    sigma_full = 1.4826 * mad_full
                    peak_full = float(np.max(data_full))

                    min_signal_full = max(float(self.config.LOW_SIGNAL_THRESHOLD), 6.0 * sigma_full)
                    if peak_full < min_signal_full:
                        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

                    cutoff_full = max(5.0, peak_full * self.config.NOISE_CUTOFF_PERCENT, 6.0 * sigma_full)
                    mask_full = data_full > cutoff_full
                    if np.count_nonzero(mask_full) < 5:
                        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

                    # Use sparse coordinates (argwhere) instead of full grids to save RAM/Time
                    # y_vals, x_vals are indices of pixels > threshold
                    y_vals, x_vals = np.nonzero(mask_full)

                    pixel_values = data_full[y_vals, x_vals]
                    pv64 = pixel_values.astype(np.float64, copy=False)
                    total_f = float(np.sum(pv64, dtype=np.float64))

                    if total_f == 0: return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

                    cnt_x = float(np.dot(x_vals.astype(np.float64), pv64) / total_f)
                    cnt_y = float(np.dot(y_vals.astype(np.float64), pv64) / total_f)

                    # For fallback, small beam default is reasonable
                    dev_x, dev_y = 10.0, 10.0
            
            d4s_x_px, d4s_y_px = 0.0, 0.0
            azimuth_deg = 0.0
            
            # 2. Refined Iterations with ROI Slicing
            # Only process the region around the beam
            for i in range(2):
                # Define ROI
                # Use generous margins to catch wings. 
                # ISO recommended integration width is >3 times parameter width.
                margin_factor = 12.0
                
                wx_roi = max(50, margin_factor * dev_x)
                wy_roi = max(50, margin_factor * dev_y)
                
                min_x = max(0, int(cnt_x - wx_roi))
                max_x = min(full_w, int(cnt_x + wx_roi))
                min_y = max(0, int(cnt_y - wy_roi))
                max_y = min(full_h, int(cnt_y + wy_roi))
                
                if (max_x - min_x) < 5 or (max_y - min_y) < 5:
                    break
                    
                # Slice full resolution data
                sub_data = img[min_y:max_y, min_x:max_x].astype(np.float32, copy=False)
                
                # CRITICAL FIX: Use GLOBAL baseline, not local.
                sub_data -= baseline 
                 
                # D4Sigma Accuracy Fix:
                # We need to reject background noise DO NOT assume it sums to zero.
                # However, the 15% configuration is for "Detection", not "Integration".
                # For integration, we want to include wings but exclude read noise.
                # We'll use a "Soft Gate": max(5.0, 1% of max).
                # This is typically 5-30 counts, which is safely above read noise (usually 2-3 counts)
                # but well below the 15% detection threshold (30-40 counts).
                
                integration_cutoff = max(1.0, max_val * 0.002) # 0.2% or 1.0 absolute
                # Zero out low-signal pixels in-place so moments only use valid signal.
                sub_data[sub_data <= integration_cutoff] = 0.0
                total_w, cx_local, cy_local, u_xx, u_yy = self._weighted_moments(sub_data)
                
                if total_w == 0: break
                
                # Update Global Centroid
                cnt_x = min_x + cx_local
                cnt_y = min_y + cy_local
                
                sub_h, sub_w = sub_data.shape
                x_idx = np.arange(sub_w, dtype=np.float32)
                y_idx = np.arange(sub_h, dtype=np.float32)
                sum_xy = float(y_idx @ sub_data @ x_idx)
                u_xy = (sum_xy / total_w) - (cx_local * cy_local)
                
                dev_x = np.sqrt(u_xx)
                dev_y = np.sqrt(u_yy)
                
                # Principal calculations
                diff = u_xx - u_yy
                sum_var = u_xx + u_yy
                
                phi = 0.5 * np.arctan2(2 * u_xy, diff)
                azimuth_deg = np.degrees(phi)
                
                term = np.sqrt(diff**2 + 4 * u_xy**2)
                d1 = 2 * np.sqrt(2) * np.sqrt(max(0.0, sum_var + term))
                d2 = 2 * np.sqrt(2) * np.sqrt(max(0.0, sum_var - term))
                
                if abs(azimuth_deg) <= 45.0:
                    d4s_x_px = d1
                    d4s_y_px = d2
                else:
                    d4s_x_px = d2
                    d4s_y_px = d1

            d4s_eff_px = np.sqrt(d4s_x_px * d4s_y_px)

            # Sanity clamp: reject pathological widths that span (almost) the whole sensor
            max_reasonable = 0.9 * float(min(full_w, full_h))
            if d4s_x_px > max_reasonable or d4s_y_px > max_reasonable:
                return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            
            pixel_um = virtual_pixel_size
            return (d4s_eff_px * pixel_um, d4s_x_px * pixel_um, d4s_y_px * pixel_um, azimuth_deg,
                    cnt_x, cnt_y, d4s_x_px, d4s_y_px)
            
        except Exception as e:
            print(f"ERROR in analyze_beam: {e}")
            traceback.print_exc()
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def calculate_caustic_fit(self, z_values: list, d_values: list, wavelength_nm: float) -> dict:
        """
        Fits hyperbola to extract M^2 parameters.
        Returns dict with M2, d0, z0, zR, theta, fit_z, fit_d.
        """
        try:
            z = np.array(z_values, dtype=np.float64)
            d = np.array(d_values, dtype=np.float64)
            
            if len(z) < 3:
                return {} # Not enough points
                
            y = d**2
            
            # Polyfit for d^2(z)
            coeffs = np.polyfit(z, y, 2)
            C = coeffs[0]
            B = coeffs[1]
            A = coeffs[2]
            
            if C <= 0:
                return {"error": "Invalid fit (Negative curvature)"}
                
            # Derived parameters
            z0 = -B / (2 * C)
            d0_sq = A - C * (z0**2)
            if d0_sq < 0:
                 # Geometric inconsistency, maybe noisy data. 
                 # Fallback: Vertex of parabola is negative? 
                 # Clamp to 0 effectively means d0=0
                 d0 = 0.0
            else:
                 d0 = np.sqrt(d0_sq)
            
            theta_rad = np.sqrt(C)
            theta_mrad = theta_rad * 1000.0
            
            
            if theta_rad > 0:
                zR = d0 / theta_rad
            else:
                zR = 0.0
                
            lam_mm = wavelength_nm * 1.0e-6
            m2 = (np.pi * d0 * theta_rad) / (4 * lam_mm)
            
            # Generate fit curve
            z_min = min(np.min(z), z0 - 2*zR)
            z_max = max(np.max(z), z0 + 2*zR)
            range_z = z_max - z_min
            if range_z == 0: range_z = 1.0
            fit_z = np.linspace(z_min - range_z*0.1, z_max + range_z*0.1, 100)
            
            fit_d_sq = d0**2 + (theta_rad**2) * (fit_z - z0)**2
            fit_d = np.sqrt(fit_d_sq)
            
            return {
                "success": True,
                "M2": m2,
                "d0_mm": d0,
                "z0_mm": z0,
                "zR_mm": zR,
                "theta_mrad": theta_mrad,
                "d0": d0*1000.0, # for consistency if used elsewhere
                "z0": z0,
                "div": theta_mrad,
                "zr": zR, # alias
                "fit_z": fit_z,
                "fit_d": fit_d,
                "raw_z": z,
                "raw_d": None
            }
            
        except Exception as e:
            print(f"Caustic Fit Error: {e}")
            return {"error": str(e)}
