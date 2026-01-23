import time
import numpy as np
from ORION.config import Config
from ORION.src.core.analysis import BeamAnalyzer

def test_perf():
    config = Config()
    analyzer = BeamAnalyzer(config)
    
    # Create large dummy image (12MP)
    w, h = 4000, 3000
    print(f"Creating {w}x{h} image...")
    img = np.zeros((h, w), dtype=np.uint8)
    
    # Add a beam spot
    y, x = np.indices((h, w))
    cy, cx = h//2, w//2
    sigma = 50
    gaussian = 200 * np.exp(-((x-cx)**2 + (y-cy)**2) / (2 * sigma**2))
    img = (img + gaussian).astype(np.uint8)
    
    # Run analysis
    max_val = 200
    v_px = 1.4
    
    print("Starting analysis...")
    start = time.time()
    res = analyzer.analyze_beam(img, max_val, v_px)
    end = time.time()
    
    print(f"Analyze Time: {end - start:.4f} seconds")
    print(f"Result: {res}")
    
    if end - start > 1.0:
        print("FAIL: Too slow!")
    elif res[0] == 0:
        print("FAIL: Beam not found!")
    else:
        print("PASS: Optimized!")

    print("\nStarting analysis (SMALL BEAM)...")
    # Reset image
    img = np.zeros((h, w), dtype=np.uint8)
    cy, cx = h//2, w//2
    sigma = 2 # Small beam
    # Manually create small spot
    img[cy-2:cy+3, cx-2:cx+3] = 200
    
    start = time.time()
    res = analyzer.analyze_beam(img, max_val, v_px)
    end = time.time()
    
    print(f"Small Beam Analyze Time: {end - start:.4f} seconds")
    print(f"Result: {res}")
    
    if res[0] > 0:
        print("PASS: Small beam detected!")
    else:
        print("FAIL: Small beam missed!")

    print("\nStarting analysis (LARGE BEAM - Test Clipping)...")
    img = np.zeros((h, w), dtype=np.uint8)
    cy, cx = h//2, w//2
    sigma_large = 200 # 200px sigma => 800px D4s. If guessed at 50, ROI would be 200-300px, clipping hard.
    # Create large gaussian
    y_g, x_g = np.indices((h, w))
    gaussian = 200 * np.exp(-((x_g-cx)**2 + (y_g-cy)**2) / (2 * sigma_large**2))
    img = (img + gaussian).astype(np.uint8)
    
    start = time.time()
    res = analyzer.analyze_beam(img, max_val, v_px)
    end = time.time()
    
    # Expected D4s pixels ~ 4 * sigma = 800
    expected_d4s_px = 4 * sigma_large
    measured_d4s_px = res[6] # d4_x_px
    
    print(f"Large Beam Analyze Time: {end - start:.4f} seconds")
    print(f"Measured D4s (px): {measured_d4s_px:.1f} (Expected ~{expected_d4s_px})")
    
    if abs(measured_d4s_px - expected_d4s_px) < 10:
        print("PASS: Large beam measured correctly (No Clipping)!")
    else:
        print("FAIL: Large beam clipped!") # Likely measured way smaller if clipped

if __name__ == "__main__":
    test_perf()
