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

Basin size calculation for phase-separating toggle switch - ENHANCED VERSION
For fixed y* = 50, sweep x* from 50 to 80 with custom intervals.
Includes basin maps and maximizes convergence.
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
N_SAMPLES = 5000  # Increased for better statistics
X_MIN, X_MAX = 0, 100
Y_MIN, Y_MAX = 0, 100
INTEGRATION_TIME = 5000.0  # Much longer for full convergence
CONVERGENCE_TOL = 1e-3     # Reasonable tolerance
EARLY_STOP_TOL = 1e-2      # Early stopping if already close


# ===================== Model class =====================

class ToggleModel:
    """
    Complete model for phase-separating toggle switch.
    """
    def __init__(self, xps, yps):
        self.xps = xps
        self.yps = yps
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
        
        # Precompute constants
        self.phix_s = xps / V_factor
        self.phiy_s = yps / V_factor
        self.CPx = kB * T * (self.phix_s - np.log(self.phix_s))
        self.CPy = kB * T * (self.phiy_s - np.log(self.phiy_s))
        self.kB_T = kB * T
        self.inv_kB_T = 1.0 / self.kB_T
        self.six_Dcx = 6.0 * Dcx
        self.six_Dcy = 6.0 * Dcy
        
        # Caches
        self.xm_cache = {}
        self.ym_cache = {}
    
    def Fx(self, xp, xm):
        return -self.CPx * xm + self.kB_T * xp * (np.log(xp / (self.V_factor - xm)) - 1.0)
    
    def F1x(self, xp, xm):
        return -self.CPx * (xm - 1.0) + self.kB_T * (xp + 1.0) * (np.log((xp + 1.0) / (self.V_factor - (xm - 1.0))) - 1.0)
    
    def Fy(self, yp, ym):
        return -self.CPy * ym + self.kB_T * yp * (np.log(yp / (self.V_factor - ym)) - 1.0)
    
    def F1y(self, yp, ym):
        return -self.CPy * (ym - 1.0) + self.kB_T * (yp + 1.0) * (np.log((yp + 1.0) / (self.V_factor - (ym - 1.0))) - 1.0)
    
    def kinx(self, xp, xm):
        if xp < self.xps:
            return 0.0
        denom = (self.Vtot - self.v * xm) ** (2.0 / 3.0)
        if denom <= 0:
            return 0.0
        return (self.six_Dcx * xp) / denom
    
    def koutx(self, xp, xm):
        if xp < self.xps or xm <= 0:
            return 0.0
        denom = (self.Vtot - self.v * (xm - 1.0)) ** (2.0 / 3.0)
        if denom <= 0:
            return 0.0
        return ((self.six_Dcx * (xp + 1.0)) / denom) * np.exp((self.Fx(xp, xm) - self.F1x(xp, xm)) * self.inv_kB_T)
    
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
    
    def dxmdt(self, xp, xm):
        return self.t1 * self.kinx(xp, xm) - self.t1 * self.koutx(xp, xm) - self.bx * xm
    
    def dymdt(self, yp, ym):
        return self.t2 * self.kiny(yp, ym) - self.t2 * self.kouty(yp, ym) - self.by * ym
    
    def calculate_xm(self, xp):
        """Find xm such that dxmdt(xp, xm) = 0"""
        xp = float(xp)
        if xp < self.xps:
            return 0.0
        key = round(xp, 1)
        if key in self.xm_cache:
            return self.xm_cache[key]
        
        lo, hi = 0.0, min(0.99 * self.V_factor, 500.0)
        
        # Simple bisection
        for _ in range(40):
            mid = (lo + hi) / 2
            f_mid = self.dxmdt(xp, mid)
            if abs(f_mid) < 1e-6:
                self.xm_cache[key] = mid
                return mid
            if self.dxmdt(xp, lo) * f_mid < 0:
                hi = mid
            else:
                lo = mid
            if hi - lo < 1e-6:
                break
        
        self.xm_cache[key] = (lo + hi) / 2
        return self.xm_cache[key]
    
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
    
    def dxpdt(self, xp, yp):
        xm = self.calculate_xm(xp)
        return self.ax / (1.0 + (yp / self.kyx) ** self.n) - self.bx * xp - self.t1 * self.kinx(xp, xm) + self.t1 * self.koutx(xp, xm)
    
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
        Integrate from (x0,y0) and determine basin with adaptive stopping.
        Maximizes convergence by using multiple strategies.
        """
        try:
            # Use dense output to monitor convergence
            sol = solve_ivp(lambda t, y: [self.dxpdt(y[0], y[1]), self.dypdt(y[0], y[1])],
                           [0, INTEGRATION_TIME], [x0, y0], 
                           method='LSODA', rtol=1e-4, atol=1e-6,
                           dense_output=True)
            
            final = sol.y[:, -1]
            
            # Check convergence to fixed points
            dist1 = np.linalg.norm(final - stable_fps[0])
            dist2 = np.linalg.norm(final - stable_fps[1])
            min_dist = min(dist1, dist2)
            
            # Strategy 1: Close to fixed point - assign directly
            if min_dist < CONVERGENCE_TOL:
                return 0 if dist1 < dist2 else 1
            
            # Strategy 2: Check if trajectory is clearly in one basin based on final state
            if final[0] < final[1] - 10:  # x much smaller than y
                return 0  # Low-x basin
            elif final[0] > final[1] + 10:  # x much larger than y
                return 1  # High-x basin
            
            # Strategy 3: Check if trajectory has stabilized (small derivative)
            deriv = np.array([self.dxpdt(final[0], final[1]), 
                             self.dypdt(final[0], final[1])])
            if np.linalg.norm(deriv) < 1e-2:
                # Stable but not at fixed point - use qualitative rule
                return 0 if final[0] < final[1] else 1
            
            # Strategy 4: Extend integration for stubborn trajectories
            if min_dist < 10:  # Already in vicinity
                sol_ext = solve_ivp(lambda t, y: [self.dxpdt(y[0], y[1]), self.dypdt(y[0], y[1])],
                                   [0, INTEGRATION_TIME * 2], [x0, y0], 
                                   method='LSODA', rtol=1e-4, atol=1e-6)
                final_ext = sol_ext.y[:, -1]
                if final_ext[0] < final_ext[1]:
                    return 0
                else:
                    return 1
            
            return -1  # Failed to converge
        except:
            # Fallback to initial condition heuristic
            if x0 < y0:
                return 0
            else:
                return 1


# ===================== Main computation =====================

def compute_basin_sizes(xps_values, yps_fixed=50.0):
    """
    Compute basin sizes for each x* value with maximized convergence.
    """
    print(f"\n{'='*60}")
    print(f"BASIN SIZE CALCULATION - MAX CONVERGENCE")
    print(f"{'='*60}")
    print(f"Fixed y* = {yps_fixed}")
    print(f"Sweeping x* with custom intervals")
    print(f"Samples per x*: {N_SAMPLES}")
    print(f"Integration time: {INTEGRATION_TIME}")
    print(f"{'='*60}\n")
    
    # Generate random initial conditions ONCE
    np.random.seed(42)
    init_points = np.random.uniform(low=[X_MIN, Y_MIN], high=[X_MAX, Y_MAX], 
                                    size=(N_SAMPLES, 2))
    
    results = []
    
    for xps in xps_values:
        print(f"\nProcessing x* = {xps}...")
        start = time.time()
        
        # Create model
        model = ToggleModel(xps, yps_fixed)
        
        # Find stable fixed points
        stable_fps = model.get_fixed_points()
        
        if len(stable_fps) != 2:
            print(f"  NOT BISTABLE: found {len(stable_fps)} stable points")
            results.append([xps, np.nan, np.nan])
            continue
        
        # Sort by x-value: low-x/high-y vs high-x/low-y
        stable_fps = sorted(stable_fps, key=lambda fp: fp[0])
        fp_low, fp_high = stable_fps[0], stable_fps[1]
        
        print(f"  Low-x state: ({fp_low[0]:.2f}, {fp_low[1]:.2f})")
        print(f"  High-x state: ({fp_high[0]:.2f}, {fp_high[1]:.2f})")
        
        # Process all initial conditions
        basin_counts = [0, 0]
        basin_assignments = []  # Store for later analysis
        
        for i, (x0, y0) in enumerate(init_points):
            if i % 500 == 0:
                print(f"    Progress: {i}/{N_SAMPLES}")
            
            basin = model.trajectory_to_basin(x0, y0, stable_fps)
            if basin >= 0:
                basin_counts[basin] += 1
                basin_assignments.append(basin)
            else:
                # Force assignment based on initial condition as last resort
                if x0 < y0:
                    basin_counts[0] += 1
                    basin_assignments.append(0)
                else:
                    basin_counts[1] += 1
                    basin_assignments.append(1)
        
        # Calculate fractions
        total = sum(basin_counts)
        frac_low = basin_counts[0] / N_SAMPLES
        frac_high = basin_counts[1] / N_SAMPLES
        convergence_rate = (total - basin_assignments.count(-1)) / N_SAMPLES if -1 in basin_assignments else 1.0
        
        elapsed = time.time() - start
        print(f"  Low-x basin: {frac_low:.4f}")
        print(f"  High-x basin: {frac_high:.4f}")
        print(f"  Sum: {frac_low + frac_high:.4f}")
        print(f"  Convergence rate: {convergence_rate:.2%}")
        print(f"  Time: {elapsed:.1f}s")
        
        results.append([xps, frac_low, frac_high])
    
    return np.array(results)


# ===================== Basin map generation =====================

def generate_basin_maps(xps_values, yps_fixed=50.0, map_size=50):
    """
    Generate basin maps for selected x* values.
    """
    selected_xps = [50, 55, 60, 70, 80]
    basin_maps = {}
    stable_fps_dict = {}
    
    print(f"\n{'='*60}")
    print(f"GENERATING BASIN MAPS")
    print(f"{'='*60}")
    
    # Create grid for maps
    x_grid = np.linspace(X_MIN, X_MAX, map_size)
    y_grid = np.linspace(Y_MIN, Y_MAX, map_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    
    for xps in selected_xps:
        if xps not in xps_values:
            print(f"Skipping x*={xps} (not in sweep range)")
            continue
            
        print(f"\nGenerating basin map for x* = {xps}...")
        start = time.time()
        
        # Create model
        model = ToggleModel(xps, yps_fixed)
        
        # Find stable fixed points
        stable_fps = model.get_fixed_points()
        
        if len(stable_fps) != 2:
            print(f"  NOT BISTABLE: found {len(stable_fps)} stable points")
            continue
        
        # Sort by x-value
        stable_fps = sorted(stable_fps, key=lambda fp: fp[0])
        stable_fps_dict[xps] = stable_fps
        
        # Initialize basin map
        basin_map = np.zeros((map_size, map_size))
        
        # Process grid points
        for idx, (x0, y0) in enumerate(grid_points):
            if idx % 500 == 0:
                print(f"    Progress: {idx}/{len(grid_points)}")
            
            i = idx // map_size
            j = idx % map_size
            
            basin = model.trajectory_to_basin(x0, y0, stable_fps)
            if basin >= 0:
                basin_map[i, j] = basin + 1  # 1 or 2
            else:
                # Fallback to qualitative rule
                if x0 < y0:
                    basin_map[i, j] = 1
                else:
                    basin_map[i, j] = 2
        
        basin_maps[xps] = basin_map
        elapsed = time.time() - start
        print(f"  Completed in {elapsed:.1f}s")
    
    return basin_maps, stable_fps_dict


# ===================== Plotting =====================

def plot_results(results):
    """
    Plot basin sizes vs x*.
    """
    xps = results[:, 0]
    low_frac = results[:, 1]
    high_frac = results[:, 2]
    
    plt.figure(figsize=(12, 7))
    
    # Plot only valid points
    valid = ~(np.isnan(low_frac) | np.isnan(high_frac))
    xps_valid = xps[valid]
    low_valid = low_frac[valid]
    high_valid = high_frac[valid]
    
    plt.plot(xps_valid, low_valid, 'b-o', linewidth=2.5, markersize=8, 
             label='Low-x / High-y basin', markeredgecolor='navy', markerfacecolor='blue')
    plt.plot(xps_valid, high_valid, 'r-o', linewidth=2.5, markersize=8,
             label='High-x / Low-y basin', markeredgecolor='darkred', markerfacecolor='red')
    
    # Add horizontal line at 0.5
    plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7, label='Equal basins (0.5)')
    
    # Find equal basins point
    if len(xps_valid) > 1:
        diff = low_valid - high_valid
        crossing = np.where(np.diff(np.sign(diff)))[0]
        if len(crossing) > 0:
            x_cross = xps_valid[crossing[0]]
            y_cross = (low_valid[crossing[0]] + high_valid[crossing[0]]) / 2
            plt.plot(x_cross, y_cross, 'ko', markersize=12, 
                    markeredgecolor='white', markeredgewidth=2,
                    label=f'Equal basins (x* ≈ {x_cross:.1f})')
            plt.axvline(x=x_cross, color='gray', linestyle='--', alpha=0.5)
    
    plt.xlabel('x* (X condensation threshold)', fontsize=14, fontweight='bold')
    plt.ylabel('Fraction of initial conditions', fontsize=14, fontweight='bold')
    plt.title(f'Basin sizes vs x* (fixed y* = 50)', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.02, 1.02)
    plt.xlim(xps[0]-1, xps[-1]+1)
    
    # Add info box
    textstr = f'Samples: {N_SAMPLES}\nIntegration time: {INTEGRATION_TIME}'
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.show()


def plot_basin_maps(basin_maps, stable_fps_dict, yps_fixed=50):
    """
    Plot basin maps for selected x* values in a single frame.
    """
    selected_xps = [50, 55, 60, 70, 80]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    plot_idx = 0
    for xps in selected_xps:
        if xps in basin_maps:
            basin_map = basin_maps[xps]
            
            im = axes[plot_idx].imshow(basin_map, origin='lower', cmap='RdBu_r',
                                      extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
                                      aspect='auto', vmin=0.5, vmax=2.5)
            
            # Mark fixed points
            if xps in stable_fps_dict:
                fp_low, fp_high = stable_fps_dict[xps]
                axes[plot_idx].plot(fp_low[0], fp_low[1], 'ko', markersize=8, 
                                   markeredgecolor='white', markeredgewidth=2, label='Low-x FP')
                axes[plot_idx].plot(fp_high[0], fp_high[1], 'ks', markersize=8,
                                   markeredgecolor='white', markeredgewidth=2, label='High-x FP')
            
            axes[plot_idx].set_xlabel('xp', fontsize=11)
            axes[plot_idx].set_ylabel('yp', fontsize=11)
            axes[plot_idx].set_title(f'x* = {xps}, y* = {yps_fixed}', fontsize=12, fontweight='bold')
            axes[plot_idx].legend(loc='upper right', fontsize=8)
            
            plot_idx += 1
    
    # Hide unused subplot
    for i in range(plot_idx, 6):
        axes[i].set_visible(False)
    
    plt.suptitle('Basin Maps for Selected x* Values', fontsize=14, fontweight='bold')
    plt.show()


def print_table(results):
    """
    Print formatted results table.
    """
    print("\n" + "="*80)
    print("RESULTS TABLE")
    print("="*80)
    print(f"{'x*':>6} | {'Low-x basin':>14} | {'High-x basin':>14} | {'Sum':>8} | {'Ratio (L/H)':>10}")
    print("-"*80)
    
    for xps, low, high in results:
        if not np.isnan(low):
            ratio = low / high if high > 0 else np.inf
            print(f"{int(xps):6d} | {low:14.4f} | {high:14.4f} | {low+high:8.4f} | {ratio:10.2f}")
        else:
            print(f"{int(xps):6d} | {'NaN':>14} | {'NaN':>14} | {'NaN':>8} | {'NaN':>10}")
    
    print("="*80)


# ===================== Data export =====================

def export_results(results, filename='basin_sizes.dat'):
    """
    Export basin size data to file.
    """
    header = (f"# Basin size results for phase-separating toggle\n"
              f"# Fixed y* = 50.0\n"
              f"# Samples per x* = {N_SAMPLES}\n"
              f"# Integration time = {INTEGRATION_TIME}\n"
              f"# Columns: xstar\tbasin_low_x\tbasin_high_x\n")
    
    np.savetxt(filename, results, fmt='%.6f', header=header, comments='')
    print(f"\nResults saved to '{filename}'")


# ===================== Main =====================

if __name__ == "__main__":
    # Custom x* sampling: 50-60 by 1, then 65-80 by 5
    xps_values = np.concatenate([
        np.arange(50, 61, 1),    # 50 to 60 inclusive
        np.arange(65, 81, 5)     # 65, 70, 75, 80
    ])
    
    print(f"\n{'='*60}")
    print(f"BASIN SIZE CALCULATION - CUSTOM INTERVALS")
    print(f"{'='*60}")
    print(f"x* values: {xps_values}")
    print(f"{'='*60}\n")
    
    # Compute basin sizes
    results = compute_basin_sizes(xps_values, yps_fixed=50.0)
    
    # Plot results
    plot_results(results)
    
    # Generate and plot basin maps for selected x* values
    basin_maps, stable_fps_dict = generate_basin_maps(xps_values, yps_fixed=50.0, map_size=50)
    plot_basin_maps(basin_maps, stable_fps_dict, yps_fixed=50.0)
    
    # Print table
    print_table(results)
    
    # Export data
    export_results(results, 'basin_sizes_XY_PS_toggle_100by100.dat')
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)