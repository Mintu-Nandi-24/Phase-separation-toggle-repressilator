"""
===============================================================================
Title: Phase separation as a tunable regulator of canonical gene regulatory motifs
===============================================================================

Authors:
    Priya Chakraborty#
        The Institute of Mathematical Sciences, CIT Campus, Taramani,
        Chennai 600113, India

    Mintu Nandi*,#
        Universal Biology Institute, The University of Tokyo,
        7-3-1 Hongo, Bunkyo-ku, Tokyo 113-0033, Japan

    Sandeep Choubey*
        The Institute of Mathematical Sciences, CIT Campus, Taramani,
        Chennai 600113, India
        Homi Bhabha National Institute, Training School Complex,
        Anushaktinagar, Mumbai 400094, India

* Corresponding authors
# Equal contributions


Basin size calculation for Y-ONLY phase-separating toggle switch
Fixed y* = 50, x* is irrelevant (no phase separation for X)
Computes fraction of initial conditions that go to low-x/high-y vs high-x/low-y states.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
import time
import warnings
warnings.filterwarnings('ignore')

# ===================== Global parameters =====================

# Toggle production/degradation
ax = 5.0
ay = 5.0
bx = 0.05
by = 0.05
kxy = 30.0
kyx = 30.0
n = 2.0

# Exchange scaling
t1 = 1.0
t2 = 1.0

# Volumes
v = 1e-25
Vtot = 1e-20
V_factor = Vtot / v  # total number of "molecular volumes" in the system

# Thermodynamics/diffusion
T = 305.0
R = 8.314
NA = 6.023e23
kB = R / NA

tau_x = 1.0 / bx
tau_y = 1.0 / by
tau_Dx = 0.1 * tau_x
tau_Dy = 0.1 * tau_y
Dcx = Vtot ** (2.0 / 3.0) / (6.0 * tau_Dx)
Dcy = Vtot ** (2.0 / 3.0) / (6.0 * tau_Dy)

# Initial guesses for fixed point search
INITIAL_GUESSES = [
    [0.1, 10], [10, 0.1], [1, 1], [10, 10], [20, 20],
    [20, 70], [50, 50], [70, 70], [70, 20], [100, 100]
]

# Numerical controls
FP_ATOL = 1e-3
FSOLVE_XTOL = 1e-8

# Basin calculation parameters - OPTIMIZED FOR MAX CONVERGENCE
N_SAMPLES = 10000  # High number for accurate single-point measurement
X_MIN, X_MAX = 0, 100
Y_MIN, Y_MAX = 0, 100
INTEGRATION_TIME = 5000.0  # Long for full convergence
CONVERGENCE_TOL = 1e-3


# ===================== Y-ONLY Model class =====================

class ToggleModel_YOnly:
    """
    Y-ONLY phase separation model.
    Only Y phase separates, X has no dense phase.
    """
    def __init__(self, yps):
        self.yps = yps
        self.xps = float('inf')  # X never phase separates (threshold at infinity)
        self.V_factor = V_factor
        self.kB = kB
        self.T = T
        self.Vtot = Vtot
        self.v = v
        self.Dcx = Dcx
        self.Dcy = Dcy
        self.bx = bx
        self.by = by
        self.ax = ax
        self.ay = ay
        self.kxy = kxy
        self.kyx = kyx
        self.n = n
        self.t1 = t1
        self.t2 = t2
        
        # Precompute constants for Y only
        self.phiy_s = yps / V_factor
        self.CPy = kB * T * (self.phiy_s - np.log(self.phiy_s))
        self.kB_T = kB * T
        self.inv_kB_T = 1.0 / self.kB_T
        self.six_Dcx = 6.0 * Dcx
        self.six_Dcy = 6.0 * Dcy
        
        # Caches (only for Y)
        self.ym_cache = {}
    
    # X never phase separates - no dense phase, no exchange rates
    def kinx(self, xp, xm):
        return 0.0
    
    def koutx(self, xp, xm):
        return 0.0
    
    # Y phase separation functions
    def Fy(self, yp, ym):
        return -self.CPy * ym + self.kB_T * yp * (np.log(yp / (self.V_factor - ym)) - 1.0)
    
    def F1y(self, yp, ym):
        return -self.CPy * (ym - 1.0) + self.kB_T * (yp + 1.0) * (np.log((yp + 1.0) / (self.V_factor - (ym - 1.0))) - 1.0)
    
    def kiny(self, yp, ym):
        if yp < self.yps:
            return 0.0
        denom = (self.Vtot - self.v * ym) ** (2.0 / 3.0)
        if denom <= 0:
            return 0.0
        return (self.six_Dcy * yp) / denom
    
    def kouty(self, yp, ym):
        if yp < self.yps or ym <= 0:
            return 0.0
        denom = (self.Vtot - self.v * (ym - 1.0)) ** (2.0 / 3.0)
        if denom <= 0:
            return 0.0
        return ((self.six_Dcy * (yp + 1.0)) / denom) * np.exp((self.Fy(yp, ym) - self.F1y(yp, ym)) * self.inv_kB_T)
    
    def dymdt(self, yp, ym):
        return self.t2 * self.kiny(yp, ym) - self.t2 * self.kouty(yp, ym) - self.by * ym
    
    def calculate_ym(self, yp):
        """Find ym such that dymdt(yp, ym) = 0"""
        yp = float(yp)
        if yp < self.yps:
            return 0.0
        key = round(yp, 1)
        if key in self.ym_cache:
            return self.ym_cache[key]
        
        lo, hi = 0.0, min(0.99 * self.V_factor, 500.0)
        
        for _ in range(40):
            mid = (lo + hi) / 2
            f_mid = self.dymdt(yp, mid)
            if abs(f_mid) < 1e-6:
                self.ym_cache[key] = mid
                return mid
            if self.dymdt(yp, lo) * f_mid < 0:
                hi = mid
            else:
                lo = mid
            if hi - lo < 1e-6:
                break
        
        self.ym_cache[key] = (lo + hi) / 2
        return self.ym_cache[key]
    
    # Dilute dynamics
    def dxpdt(self, xp, yp):
        # No X phase separation terms
        return self.ax / (1.0 + (yp / self.kyx) ** self.n) - self.bx * xp
    
    def dypdt(self, xp, yp):
        ym = self.calculate_ym(yp)
        return self.ay / (1.0 + (xp / self.kxy) ** self.n) - self.by * yp - self.t2 * self.kiny(yp, ym) + self.t2 * self.kouty(yp, ym)
    
    def get_fixed_points(self):
        """Find all fixed points and identify the two stable ones"""
        fps = []
        
        for g in INITIAL_GUESSES:
            try:
                sol = fsolve(lambda X: [self.dxpdt(X[0], X[1]), self.dypdt(X[0], X[1])],
                             x0=np.array(g, dtype=float), xtol=FSOLVE_XTOL, maxfev=2000)
                
                if sol[0] < 0 or sol[1] < 0:
                    continue
                
                res = np.array([self.dxpdt(sol[0], sol[1]), self.dypdt(sol[0], sol[1])])
                if np.linalg.norm(res) > 1e-4:
                    continue
                
                if not any(np.allclose(sol, fp, atol=FP_ATOL) for fp in fps):
                    fps.append(sol)
            except:
                continue
        
        # Classify stability
        stable_fps = []
        for fp in fps:
            xp, yp = fp
            eps = 1e-4
            
            # Numerical Jacobian
            f0 = self.dxpdt(xp, yp)
            g0 = self.dypdt(xp, yp)
            
            J = np.array([
                [(self.dxpdt(xp + eps, yp) - f0) / eps,
                 (self.dxpdt(xp, yp + eps) - f0) / eps],
                [(self.dypdt(xp + eps, yp) - g0) / eps,
                 (self.dypdt(xp, yp + eps) - g0) / eps]
            ])
            
            eig = np.linalg.eigvals(J)
            if np.all(np.real(eig) < 0):
                stable_fps.append(fp)
        
        return stable_fps
    
    def trajectory_to_basin(self, x0, y0, stable_fps):
        """
        Integrate from (x0,y0) and determine basin.
        """
        try:
            sol = solve_ivp(lambda t, y: [self.dxpdt(y[0], y[1]), self.dypdt(y[0], y[1])],
                           [0, INTEGRATION_TIME], [x0, y0], 
                           method='LSODA', rtol=1e-4, atol=1e-6)
            
            final = sol.y[:, -1]
            
            # Check convergence to fixed points
            dist1 = np.linalg.norm(final - stable_fps[0])
            dist2 = np.linalg.norm(final - stable_fps[1])
            min_dist = min(dist1, dist2)
            
            # Strategy 1: Close to fixed point
            if min_dist < CONVERGENCE_TOL:
                return 0 if dist1 < dist2 else 1
            
            # Strategy 2: Qualitative rule based on final state
            if final[0] < final[1] - 10:  # x much smaller than y
                return 0
            elif final[0] > final[1] + 10:  # x much larger than y
                return 1
            
            # Strategy 3: Check derivative
            deriv = np.array([self.dxpdt(final[0], final[1]), 
                             self.dypdt(final[0], final[1])])
            if np.linalg.norm(deriv) < 1e-2:
                return 0 if final[0] < final[1] else 1
            
            # Strategy 4: Extended integration
            sol_ext = solve_ivp(lambda t, y: [self.dxpdt(y[0], y[1]), self.dypdt(y[0], y[1])],
                               [0, INTEGRATION_TIME * 2], [x0, y0], 
                               method='LSODA', rtol=1e-4, atol=1e-6)
            final_ext = sol_ext.y[:, -1]
            
            if final_ext[0] < final_ext[1]:
                return 0
            else:
                return 1
                
        except:
            # Fallback to initial condition heuristic
            if x0 < y0:
                return 0
            else:
                return 1
    
    def generate_basin_map(self, map_size=100):
        """
        Generate a basin map for visualization.
        """
        print(f"\nGenerating basin map ({map_size}×{map_size})...")
        
        # Create grid
        x_grid = np.linspace(X_MIN, X_MAX, map_size)
        y_grid = np.linspace(Y_MIN, Y_MAX, map_size)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Find stable fixed points
        stable_fps = self.get_fixed_points()
        if len(stable_fps) != 2:
            print("  WARNING: Could not find two stable fixed points")
            return None, None
        
        stable_fps = sorted(stable_fps, key=lambda fp: fp[0])
        
        # Initialize basin map
        basin_map = np.zeros((map_size, map_size))
        
        # Process grid points
        for i in range(map_size):
            if i % 20 == 0:
                print(f"    Row {i}/{map_size}")
            for j in range(map_size):
                x0, y0 = X[i, j], Y[i, j]
                basin = self.trajectory_to_basin(x0, y0, stable_fps)
                basin_map[i, j] = basin + 1  # 1 or 2
        
        return basin_map, stable_fps


# ===================== Single-point basin calculation =====================

def compute_single_basin_size(yps_fixed=50.0):
    """
    Compute basin fractions for a single threshold value.
    """
    print(f"\n{'='*60}")
    print(f"Y-ONLY PHASE SEPARATION - SINGLE POINT CALCULATION")
    print(f"{'='*60}")
    print(f"Fixed y* = {yps_fixed}")
    print(f"Samples: {N_SAMPLES}")
    print(f"Integration time: {INTEGRATION_TIME}")
    print(f"{'='*60}\n")
    
    # Generate random initial conditions
    np.random.seed(42)
    init_points = np.random.uniform(low=[X_MIN, Y_MIN], high=[X_MAX, Y_MAX], 
                                    size=(N_SAMPLES, 2))
    
    start = time.time()
    
    # Create model
    model = ToggleModel_YOnly(yps_fixed)
    
    # Find stable fixed points
    stable_fps = model.get_fixed_points()
    
    if len(stable_fps) != 2:
        print(f"ERROR: Found {len(stable_fps)} stable fixed points (need 2)")
        return None, None, None
    
    # Sort by x-value
    stable_fps = sorted(stable_fps, key=lambda fp: fp[0])
    fp_low, fp_high = stable_fps[0], stable_fps[1]
    
    print(f"\nStable fixed points:")
    print(f"  Low-x state:  ({fp_low[0]:.2f}, {fp_low[1]:.2f})")
    print(f"  High-x state: ({fp_high[0]:.2f}, {fp_high[1]:.2f})")
    
    # Process all initial conditions
    basin_counts = [0, 0]
    
    for i, (x0, y0) in enumerate(init_points):
        if i % 1000 == 0:
            print(f"  Progress: {i}/{N_SAMPLES}")
        
        basin = model.trajectory_to_basin(x0, y0, stable_fps)
        if basin >= 0:
            basin_counts[basin] += 1
        else:
            # Force assignment as last resort
            if x0 < y0:
                basin_counts[0] += 1
            else:
                basin_counts[1] += 1
    
    # Calculate fractions
    total = sum(basin_counts)
    frac_low = basin_counts[0] / N_SAMPLES
    frac_high = basin_counts[1] / N_SAMPLES
    
    elapsed = time.time() - start
    
    print(f"\nResults:")
    print(f"  Low-x basin:  {frac_low:.4f} ({basin_counts[0]} points)")
    print(f"  High-x basin: {frac_high:.4f} ({basin_counts[1]} points)")
    print(f"  Sum: {frac_low + frac_high:.4f}")
    print(f"  Time: {elapsed:.1f}s")
    
    return model, stable_fps, (frac_low, frac_high)


# ===================== Plotting =====================

def plot_basin_map(model, stable_fps, fractions, yps_fixed=50.0, map_size=100):
    """
    Plot basin map with results.
    """
    # Generate basin map
    basin_map, _ = model.generate_basin_map(map_size=map_size)
    
    if basin_map is None:
        return
    
    frac_low, frac_high = fractions
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Basin map
    im = ax1.imshow(basin_map, origin='lower', cmap='RdBu_r',
                   extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
                   aspect='auto', vmin=0.5, vmax=2.5)
    
    # Mark fixed points
    fp_low, fp_high = sorted(stable_fps, key=lambda fp: fp[0])
    ax1.plot(fp_low[0], fp_low[1], 'ko', markersize=10, 
            markeredgecolor='white', markeredgewidth=2, label='Low-x FP')
    ax1.plot(fp_high[0], fp_high[1], 'ks', markersize=10,
            markeredgecolor='white', markeredgewidth=2, label='High-x FP')
    
    ax1.set_xlabel('xp', fontsize=12)
    ax1.set_ylabel('yp', fontsize=12)
    ax1.set_title(f'Y-Only Phase Separation (y* = {yps_fixed})\nBasin Map', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    plt.colorbar(im, ax=ax1, label='Basin (1=Low-x, 2=High-x)')
    
    # Right: Pie chart of basin fractions
    colors = ['blue', 'red']
    labels = [f'Low-x/High-y\n({frac_low:.1%})', 
              f'High-x/Low-y\n({frac_high:.1%})']
    
    ax2.pie([frac_low, frac_high], labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12})
    ax2.set_title('Basin Fractions', fontsize=14, fontweight='bold')
    
    # Add info text
    info_text = (f"y* = {yps_fixed}\n"
                 f"Samples: {N_SAMPLES}\n"
                 f"Integration time: {INTEGRATION_TIME}")
    ax2.text(0.5, -0.2, info_text, transform=ax2.transAxes,
             ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.show()


def print_summary(fractions):
    """
    Print formatted summary.
    """
    frac_low, frac_high = fractions
    
    print("\n" + "="*60)
    print("SUMMARY - Y-ONLY PHASE SEPARATION")
    print("="*60)
    print(f"Fixed y* = 50.0")
    print(f"Samples: {N_SAMPLES}")
    print("-"*60)
    print(f"Low-x/High-y basin:  {frac_low:.4f} ({frac_low:.1%})")
    print(f"High-x/Low-y basin: {frac_high:.4f} ({frac_high:.1%})")
    print(f"Sum:                 {frac_low + frac_high:.4f}")
    print(f"Ratio (L/H):         {frac_low/frac_high:.4f}")
    print("="*60)


def export_results(fractions, filename='y_only_basin_results.dat'):
    """
    Export results to file.
    """
    frac_low, frac_high = fractions
    
    header = (f"# Y-ONLY phase separation results\n"
              f"# Fixed y* = 50.0\n"
              f"# Samples = {N_SAMPLES}\n"
              f"# Integration time = {INTEGRATION_TIME}\n"
              f"# Columns: parameter\tvalue\n")
    
    data = np.array([
        [1, frac_low],
        [2, frac_high],
        [3, frac_low + frac_high],
        [4, frac_low/frac_high]
    ])
    
    np.savetxt(filename, data, fmt='%.6f', header=header, comments='')
    print(f"\nResults saved to '{filename}'")


# ===================== Main =====================

if __name__ == "__main__":
    # Fixed y* threshold
    yps_fixed = 50.0
    
    # Compute basin sizes for single point
    model, stable_fps, fractions = compute_single_basin_size(yps_fixed)
    
    if fractions is not None:
        # Plot basin map and pie chart
        plot_basin_map(model, stable_fps, fractions, yps_fixed, map_size=80)
        
        # Print summary
        print_summary(fractions)
        
        # Export results
        export_results(fractions, 'basin_sizes_Y_PS_toggle_100by100.dat')
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)