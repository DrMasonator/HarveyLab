import numpy as np
import traceback
from ORION.config import Config

class BeamAnalyzer:
    def __init__(self, config: Config):
        self.config = config

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
            data_coarse = img[::stride, ::stride].astype(np.float64)
            baseline = np.min(data_coarse)
            data_coarse -= baseline
            
            cutoff = max(5.0, max_val * self.config.NOISE_CUTOFF_PERCENT)
            mask_coarse = data_coarse > cutoff
            
            found_coarse = np.sum(mask_coarse) >= 5
            
            if found_coarse:
                sh, sw = data_coarse.shape
                y_grid_s, x_grid_s = np.indices((sh, sw))
                
                total_s = np.sum(data_coarse[mask_coarse])
                # Safe check for div by zero although found_coarse implies > 0
                if total_s == 0: return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                
                cx_s = np.sum(x_grid_s[mask_coarse] * data_coarse[mask_coarse]) / total_s
                cy_s = np.sum(y_grid_s[mask_coarse] * data_coarse[mask_coarse]) / total_s
                
                cnt_x = cx_s * stride
                cnt_y = cy_s * stride
                
                # Estimate width from coarse data to set proper initial ROI
                dx_s = x_grid_s - cx_s
                dy_s = y_grid_s - cy_s
                
                weights_s = data_coarse * mask_coarse # Apply mask
                # Note: This sigma is calculated ONLY on pixels > cutoff. 
                # Real Gaussian wings extend much further. 
                # This underestimates sigma by ~30-50% depending on cutoff.
                u_xx_s = np.sum(weights_s * dx_s**2) / total_s
                u_yy_s = np.sum(weights_s * dy_s**2) / total_s
                
                # Apply 4.0x safety factor because the mask cuts off wings drastically
                dev_x = np.sqrt(u_xx_s) * stride * 4.0
                dev_y = np.sqrt(u_yy_s) * stride * 4.0
                
            else:
                # FALLBACK: Small Beam Detection
                # If the beam is tiny (e.g. < 4px), stride=4 might skip it.
                # Check full image, but use sparse calculation to remain fast.
                data_full = img.astype(np.float64)
                # Recalculate baseline on full img to be safe
                base_full = np.min(data_full)
                data_full -= base_full
                
                mask_full = data_full > cutoff
                if np.sum(mask_full) < 5:
                    return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                
                # Use sparse coordinates (argwhere) instead of full grids to save RAM/Time
                # y_vals, x_vals are indices of pixels > threshold
                coords = np.argwhere(mask_full)
                y_vals = coords[:, 0]
                x_vals = coords[:, 1]
                
                pixel_values = data_full[y_vals, x_vals]
                total_f = np.sum(pixel_values)
                
                if total_f == 0: return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                
                cnt_x = np.sum(x_vals * pixel_values) / total_f
                cnt_y = np.sum(y_vals * pixel_values) / total_f

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
                margin_factor = 8.0
                
                wx_roi = max(50, margin_factor * dev_x)
                wy_roi = max(50, margin_factor * dev_y)
                
                min_x = max(0, int(cnt_x - wx_roi))
                max_x = min(full_w, int(cnt_x + wx_roi))
                min_y = max(0, int(cnt_y - wy_roi))
                max_y = min(full_h, int(cnt_y + wy_roi))
                
                if (max_x - min_x) < 5 or (max_y - min_y) < 5:
                    break
                    
                # Slice full resolution data
                sub_data = img[min_y:max_y, min_x:max_x].astype(np.float64)
                
                # CRITICAL FIX: Use GLOBAL baseline, not local.
                sub_data -= baseline 
                 
                # D4Sigma Accuracy Fix:
                # We need to reject background noise DO NOT assume it sums to zero.
                # However, the 15% configuration is for "Detection", not "Integration".
                # For integration, we want to include wings but exclude read noise.
                # We'll use a "Soft Gate": max(5.0, 1% of max).
                # This is typically 5-30 counts, which is safely above read noise (usually 2-3 counts)
                # but well below the 15% detection threshold (30-40 counts).
                
                integration_cutoff = max(5.0, max_val * 0.01) # 1% or 5.0 absolute
                sub_mask = sub_data > integration_cutoff
                
                weights = sub_data * sub_mask
                
                total_w = np.sum(weights)
                
                if total_w == 0: break
                
                # Local grids
                sub_h, sub_w = sub_data.shape
                loc_y, loc_x = np.indices((sub_h, sub_w))
                
                # Local moments
                cx_local = np.sum(loc_x * weights) / total_w
                cy_local = np.sum(loc_y * weights) / total_w
                
                # Update Global Centroid
                cnt_x = min_x + cx_local
                cnt_y = min_y + cy_local
                
                # Second moments (Centered)
                dx = loc_x - cx_local
                dy = loc_y - cy_local
                
                u_xx = np.sum(weights * dx**2) / total_w
                u_yy = np.sum(weights * dy**2) / total_w
                u_xy = np.sum(weights * dx * dy) / total_w
                
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
