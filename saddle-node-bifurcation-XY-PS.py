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

Bifurcation (phase) diagram in the (xps, yps) plane for the phase-separating toggle
using dilute variables (xp, yp) with quasi-steady dense pools xm(xp), ym(yp).

What it computes:
- For each (xps, yps): find all fixed points of the reduced 2D system
    dxp/dt = 0, dyp/dt = 0
  where xm and ym are obtained from dxm/dt=0 and dym/dt=0 (brentq).
- Classify fixed points by Jacobian eigenvalues (central differences).
- Mark bistability if: (#stable >= 2 and #saddle >= 1)
- Plot a bistability map and its boundary contour.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, brentq

# ===================== Global parameters (set once) =====================

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

# Initial guesses for fixed point search (keep as in your script, can expand)
INITIAL_GUESSES = [
    [0.1, 10], [10, 0.1], [1, 1], [10, 10], [20, 20],
    [20, 70], [50, 50], [70, 70], [70, 20], [100, 100]
]

# Numerical controls
FP_ATOL = 1e-3        # duplicate fixed-point merge tolerance
FSOLVE_XTOL = 1e-10   # fixed point solve tolerance
JAC_EPS = 1e-5        # central-difference step for Jacobian
DET_TOL = 1e-5        # optional: "near-fold" threshold on |det J|


# ===================== Model components (depend on xps,yps via CPx, CPy) =====================

def make_model_for_thresholds(xps, yps):
    """
    Returns closures:
      calculate_xm(xp), calculate_ym(yp),
      dxpdt(xp,yp), dypdt(xp,yp),
      jacobian(xp,yp)
    with CPx, CPy set consistently from xps,yps.
    """

    # Derived thermodynamic parameters from thresholds (as in your code)
    phix_s = xps / V_factor
    phiy_s = yps / V_factor
    CPx = kB * T * (phix_s - np.log(phix_s))
    CPy = kB * T * (phiy_s - np.log(phiy_s))

    # ---------- Free energies ----------
    def Fx(xp, xm):
        # NOTE: requires xp>0; we guard where needed
        return -CPx * xm + kB * T * xp * (np.log(xp / ((V_factor) - xm)) - 1.0)

    def F1x(xp, xm):
        return -CPx * (xm - 1.0) + kB * T * (xp + 1.0) * (np.log((xp + 1.0) / (V_factor - (xm - 1.0))) - 1.0)

    def Fy(yp, ym):
        return -CPy * ym + kB * T * yp * (np.log(yp / ((V_factor) - ym)) - 1.0)

    def F1y(yp, ym):
        return -CPy * (ym - 1.0) + kB * T * (yp + 1.0) * (np.log((yp + 1.0) / (V_factor - (ym - 1.0))) - 1.0)

    # ---------- Exchange rates ----------
    def kinx(xp, xm):
        if xp < xps:
            return 0.0
        # diffusion-limited, as in your code
        denom = (Vtot - v * xm) ** (2.0 / 3.0)
        if denom <= 0:
            return 0.0
        return (6.0 * Dcx * xp) / denom

    def koutx(xp, xm):
        if xp < xps or xm < 0:
            return 0.0
        denom = (Vtot - v * (xm - 1.0)) ** (2.0 / 3.0)
        if denom <= 0:
            return 0.0
        # detailed balance form
        return ((6.0 * Dcx * (xp + 1.0)) / denom) * np.exp((Fx(xp, xm) - F1x(xp, xm)) / (kB * T))

    def kiny(yp, ym):
        if yp < yps:
            return 0.0
        denom = (Vtot - v * ym) ** (2.0 / 3.0)
        if denom <= 0:
            return 0.0
        return (6.0 * Dcy * yp) / denom

    def kouty(yp, ym):
        if yp < yps or ym < 0:
            return 0.0
        denom = (Vtot - v * (ym - 1.0)) ** (2.0 / 3.0)
        if denom <= 0:
            return 0.0
        return ((6.0 * Dcy * (yp + 1.0)) / denom) * np.exp((Fy(yp, ym) - F1y(yp, ym)) / (kB * T))

    # ---------- Dense pool quasi-steady closure ----------
    # Use small per-call caches to avoid repeated root solves in Jacobian evaluations
    xm_cache = {}
    ym_cache = {}

    def dxmdt(xp, xm):
        return t1 * kinx(xp, xm) - t1 * koutx(xp, xm) - bx * xm

    def dymdt(yp, ym):
        return t2 * kiny(yp, ym) - t2 * kouty(yp, ym) - by * ym

    def brentq_with_scan(fun, lo, hi, nscan=60):
        xs = np.linspace(lo, hi, nscan)
        fs = np.array([fun(x) for x in xs], dtype=float)
        # look for sign change
        for i in range(nscan-1):
            if not np.isfinite(fs[i]) or not np.isfinite(fs[i+1]):
                continue
            if fs[i] == 0:
                return xs[i]
            if fs[i]*fs[i+1] < 0:
                return brentq(fun, xs[i], xs[i+1], maxiter=200)
        raise ValueError("no bracket")
        

    # def calculate_xm(xp):
    #     xp = float(xp)
    #     if xp < xps:
    #         return 0.0
    #     if xp in xm_cache:
    #         return xm_cache[xp]
    #     # bracket: xm must be < V_factor to keep (V_factor - xm) positive
    #     lo = 0.0
    #     hi = min(0.999 * V_factor, 500.0)  # keep your 500 cap but prevent volume blow-up
    #     try:
    #         root = brentq(lambda xm: dxmdt(xp, xm), lo, hi, maxiter=200)
    #     except ValueError:
    #         root = 0.0
    #     xm_cache[xp] = root
    #     return root
    
    def calculate_xm(xp):
        xp = float(xp)
        if xp < xps:
            return 0.0
        key = float(xp) #round(xp, 2)   # IMPORTANT: quantize key for better cache hits
        if key in xm_cache:
            return xm_cache[key]
        
        lo = 0.0
        hi = min(0.999 * V_factor, 500.0)
        
        def fun(xm):
            return dxmdt(xp, xm)
        
        try:
            root = brentq_with_scan(fun, lo, hi, nscan=60)
        except ValueError:
            root = 0.0

        xm_cache[key] = root
        return root
    



    # def calculate_ym(yp):
    #     yp = float(yp)
    #     if yp < yps:
    #         return 0.0
    #     if yp in ym_cache:
    #         return ym_cache[yp]
    #     lo = 0.0
    #     hi = min(0.999 * V_factor, 500.0)
    #     try:
    #         root = brentq(lambda ym: dymdt(yp, ym), lo, hi, maxiter=200)
    #     except ValueError:
    #         root = 0.0
    #     ym_cache[yp] = root
    #     return root
    
    def calculate_ym(yp):
        yp = float(yp)
        if yp < yps:
            return 0.0
        key = float(yp) #round(yp, 2)   # IMPORTANT: quantize key for better cache hits
        if key in ym_cache:
            return ym_cache[key]
        
        lo = 0.0
        hi = min(0.999 * V_factor, 500.0)
        
        def fun(ym):
            return dymdt(yp, ym)
        
        try:
            root = brentq_with_scan(fun, lo, hi, nscan=60)
        except ValueError:
            root = 0.0

        ym_cache[key] = root
        return root


    # ---------- Dilute dynamics (2D system) ----------
    def dxpdt(xp, yp):
        xm = calculate_xm(xp)
        return ax * (1.0 / (1.0 + (yp / kyx) ** n)) - bx * xp - t1 * kinx(xp, xm) + t1 * koutx(xp, xm)

    def dypdt(xp, yp):
        ym = calculate_ym(yp)
        return ay * (1.0 / (1.0 + (xp / kxy) ** n)) - by * yp - t2 * kiny(yp, ym) + t2 * kouty(yp, ym)

# ---------- Jacobian by central differences (robust) ----------
    def jacobian(xp, yp, eps=JAC_EPS):
        xp = float(xp); yp = float(yp)
        # Guard against negative evaluations when using central differences
        xp_p = max(xp + eps, 0.0)
        xp_m = max(xp - eps, 0.0)
        yp_p = max(yp + eps, 0.0)
        yp_m = max(yp - eps, 0.0)

        F_xp_p = dxpdt(xp_p, yp)
        F_xp_m = dxpdt(xp_m, yp)
        F_yp_p = dxpdt(xp, yp_p)
        F_yp_m = dxpdt(xp, yp_m)

        G_xp_p = dypdt(xp_p, yp)
        G_xp_m = dypdt(xp_m, yp)
        G_yp_p = dypdt(xp, yp_p)
        G_yp_m = dypdt(xp, yp_m)

        dFdx = (F_xp_p - F_xp_m) / (xp_p - xp_m) if xp_p != xp_m else 0.0
        dFdy = (F_yp_p - F_yp_m) / (yp_p - yp_m) if yp_p != yp_m else 0.0
        dGdx = (G_xp_p - G_xp_m) / (xp_p - xp_m) if xp_p != xp_m else 0.0
        dGdy = (G_yp_p - G_yp_m) / (yp_p - yp_m) if yp_p != yp_m else 0.0

        return np.array([[dFdx, dFdy],
                          [dGdx, dGdy]], dtype=float)
    
    return dxpdt, dypdt, jacobian

# ===================== Fixed points + stability classification =====================

# def find_fixed_points(dxpdt, dypdt, guesses=INITIAL_GUESSES):
#     fps = []
#     for g in guesses:
#         sol = fsolve(lambda X: [dxpdt(X[0], X[1]), dypdt(X[0], X[1])],
#                       x0=np.array(g, dtype=float), xtol=FSOLVE_XTOL, maxfev=2000)
#         if sol[0] < 0 or sol[1] < 0:
#             continue
#         if not any(np.allclose(sol, fp, atol=FP_ATOL) for fp in fps):
#             fps.append(sol)
#     return fps

# def find_fixed_points(dxpdt, dypdt, guesses=INITIAL_GUESSES):
#     RES_TOL = 1e-8  # can tune

#     fps = []
#     for g in guesses:
#         sol = fsolve(lambda X: [dxpdt(X[0], X[1]), dypdt(X[0], X[1])],
#                       x0=np.array(g, dtype=float), xtol=FSOLVE_XTOL, maxfev=2000)

#         # reject non-physical
#         if sol[0] < 0 or sol[1] < 0:
#             continue

#         # >>> ADD THIS BLOCK HERE (right here) <<<
#         res = np.array([dxpdt(sol[0], sol[1]), dypdt(sol[0], sol[1])], dtype=float)
#         if np.linalg.norm(res, ord=2) > RES_TOL:
#             continue
#         # >>> END ADD <<<

#         # deduplicate
#         if not any(np.allclose(sol, fp, atol=FP_ATOL) for fp in fps):
#             fps.append(sol)

#     return fps

def xp_nullcline_f(yp, dxpdt, x_min=0.0, x_max=120.0, nscan=200):
    """
    Returns xp such that dxpdt(xp, yp) = 0.
    Uses bracket scan + brentq. Returns np.nan if not found.
    """
    xs = np.linspace(x_min, x_max, nscan)
    fs = np.array([dxpdt(x, yp) for x in xs], dtype=float)

    for i in range(nscan - 1):
        if not np.isfinite(fs[i]) or not np.isfinite(fs[i+1]):
            continue
        if fs[i] == 0:
            return xs[i]
        if fs[i] * fs[i+1] < 0:
            return brentq(lambda x: dxpdt(x, yp), xs[i], xs[i+1], maxiter=200)
    return np.nan


def yp_nullcline_f(xp, dypdt, y_min=0.0, y_max=120.0, nscan=200):
    """
    Returns yp such that dypdt(xp, yp) = 0.
    Uses bracket scan + brentq. Returns np.nan if not found.
    """
    ys = np.linspace(y_min, y_max, nscan)
    fs = np.array([dypdt(xp, y) for y in ys], dtype=float)

    for i in range(nscan - 1):
        if not np.isfinite(fs[i]) or not np.isfinite(fs[i+1]):
            continue
        if fs[i] == 0:
            return ys[i]
        if fs[i] * fs[i+1] < 0:
            return brentq(lambda y: dypdt(xp, y), ys[i], ys[i+1], maxiter=200)
    return np.nan

def find_fixed_points(dxpdt, dypdt, guesses):
    """
    Find fixed points of the 2D system and keep only those consistent with nullcline intersections.
    """
    nullcline_threshold=1e-6
    x_bounds=(0.0, 120.0)
    y_bounds=(0.0, 120.0)
    
    fps = []

    for g in guesses:
        RES_TOL = 1e-8
        sol = fsolve(lambda X: [dxpdt(X[0], X[1]), dypdt(X[0], X[1])],
                     x0=np.array(g, dtype=float),
                     xtol=FSOLVE_XTOL, maxfev=2000)
        # if not any(np.allclose(sol, fp, atol=1e-3) for fp in fps):  # Avoid duplicates
        #     sol = sol

        # reject non-physical
        if sol[0] < 0 or sol[1] < 0:
            continue

        # residual check (must satisfy ODEs)
        res = np.array([dxpdt(sol[0], sol[1]), dypdt(sol[0], sol[1])], dtype=float)
        if np.linalg.norm(res, ord=2) > RES_TOL:
            continue

        # deduplicate
        if any(np.allclose(sol, fp, atol=FP_ATOL) for fp in fps):
            continue

        # ---------- Nullcline consistency filter ----------
        xp_star, yp_star = float(sol[0]), float(sol[1])

        xp_null = xp_nullcline_f(yp_star, dxpdt,
                                 x_min=x_bounds[0], x_max=x_bounds[1])
        yp_null = yp_nullcline_f(xp_star, dypdt,
                                 y_min=y_bounds[0], y_max=y_bounds[1])

        # if nullcline solver failed, reject
        if not np.isfinite(xp_null) or not np.isfinite(yp_null):
            continue

        # keep only if close to both nullclines
        if (abs(xp_star - xp_null) < nullcline_threshold and
            abs(yp_star - yp_null) < nullcline_threshold):
            fps.append(sol)

    return fps

#_------------------------------------------------------------------------------

def spectrum_plot_at(xps0, yps0, guesses=INITIAL_GUESSES):
    """
    Returns:
      fps: list of fixed points [xp, yp]
      eigs: list of eigenvalue arrays (len=2) corresponding to each fp
    """
    dxpdt, dypdt, jac = make_model_for_thresholds(xps0, yps0)
    fps = find_fixed_points(dxpdt, dypdt, guesses=guesses)

    eigs = []
    for fp in fps:
        eigs.append(np.linalg.eigvals(jac(fp[0], fp[1])))

    return fps, eigs


#_-------------------------------------------

def classify_from_J(J, tol=1e-10):
    tr = float(np.trace(J))
    det = float(np.linalg.det(J))

    # saddle in 2D: det < 0
    if det < -tol:
        return "saddle"

    # stable in 2D: det > 0 and trace < 0
    if det > tol and tr < -tol:
        return "stable"

    # unstable (source) in 2D: det > 0 and trace > 0
    if det > tol and tr > tol:
        return "unstable"

    # near-fold / near-degenerate
    return "near"



# def classify_fixed_points(fps, jacobian):
#     n_stable = 0
#     n_saddle = 0
#     min_abs_detJ = np.inf

#     for fp in fps:
#         J = jacobian(fp[0], fp[1])
#         eig = np.linalg.eigvals(J)
#         detJ = np.linalg.det(J)
#         min_abs_detJ = min(min_abs_detJ, abs(detJ))

#         # stable if Re(lambda)<0 for both eigenvalues
#         if np.all(np.real(eig) < 0):
#             n_stable += 1
#         # saddle if product of Re parts is negative (typical in 2D)
#         elif (np.real(eig[0]) * np.real(eig[1]) < 0):
#             n_saddle += 1

#     bistable = (n_stable >= 2 and n_saddle >= 1)
#     return bistable, n_stable, n_saddle, min_abs_detJ

def classify_fixed_points(fps, jacobian, tol=1e-10):
    n_stable = 0
    n_saddle = 0
    n_unstable = 0
    n_near = 0
    min_abs_detJ = np.inf

    for fp in fps:
        J = jacobian(fp[0], fp[1])
        detJ = float(np.linalg.det(J))
        min_abs_detJ = min(min_abs_detJ, abs(detJ))

        tp = classify_from_J(J, tol=tol)
        if tp == "stable":
            n_stable += 1
        elif tp == "saddle":
            n_saddle += 1
        elif tp == "unstable":
            n_unstable += 1
        else:
            n_near += 1

    # Your original rule (toggle-like): classical bistability requires 2 stable + 1 saddle
    bistable = (n_stable >= 2 and n_saddle >= 1)

    return bistable, n_stable, n_saddle, min_abs_detJ, n_unstable, n_near

#_---------------------------------------------

def analyze_threshold_pair(xps, yps):
    dxpdt, dypdt, jac = make_model_for_thresholds(xps, yps)
    fps = find_fixed_points(dxpdt, dypdt)
    return classify_fixed_points(fps, jac)


# ===================== Bifurcation diagram scan =====================

# def scan_bifurcation(xps_values, yps_values):
#     B = np.zeros((len(yps_values), len(xps_values)), dtype=int)  # 1=bistable
#     S = np.zeros_like(B, dtype=int)  # #stable
#     D = np.zeros_like(B, dtype=int)  # #saddle
#     M = np.zeros_like(B, dtype=float)  # min |detJ|

#     for iy, yps in enumerate(yps_values):
#         for ix, xps in enumerate(xps_values):
#             bistable, ns, nd, min_det = analyze_threshold_pair(xps, yps)
#             B[iy, ix] = 1 if bistable else 0
#             S[iy, ix] = ns
#             D[iy, ix] = nd
#             M[iy, ix] = min_det
            
#             print("------------------------------------------------------------")  
#             # print("Fixed points", fps)
#             print(f"(xps,yps)=({xps:.2f},{yps:.2f})  bistable={bistable}  stable={ns} saddle={nd}  min|detJ|={min_det:.3e}")
#     return B, S, D, M

def scan_bifurcation(xps_values, yps_values):
    B = np.zeros((len(yps_values), len(xps_values)), dtype=int)
    S = np.zeros_like(B, dtype=int)
    D = np.zeros_like(B, dtype=int)
    M = np.zeros_like(B, dtype=float)

    prev_row_fps = [None] * len(xps_values)

    for iy, yps in enumerate(yps_values):
        left_fps = None
        for ix, xps in enumerate(xps_values):
            dxpdt, dypdt, jac = make_model_for_thresholds(xps, yps)

            warm = []
            if left_fps:
                warm += left_fps
            if prev_row_fps[ix]:
                warm += prev_row_fps[ix]

            # build guesses = warm + default guesses
            guesses = warm + INITIAL_GUESSES

            fps = find_fixed_points(dxpdt, dypdt, guesses=guesses)
            bistable, ns, nd, min_det, nu, nn = classify_fixed_points(fps, jac)
            # bistable, ns, nd, min_det = classify_fixed_points(fps, jac)

            B[iy, ix], S[iy, ix], D[iy, ix], M[iy, ix] = int(bistable), ns, nd, min_det
            left_fps = fps
            prev_row_fps[ix] = fps

            print("------------------------------------------------------------")  
            print("Fixed points", fps)
            print(f"(xps,yps)=({xps:.2f},{yps:.2f})  bistable={bistable}  stable={ns} saddle={nd}  min|detJ|={min_det:.3e}")
            
    return B, S, D, M


# ===================== Run =====================

def boundary_cells(B):
    cells = []
    ny, nx = B.shape
    for iy in range(ny - 1):
        for ix in range(nx - 1):
            block = B[iy:iy+2, ix:ix+2]
            if np.min(block) != np.max(block):
                cells.append((iy, ix))
    return cells


def refine_cells(xps_grid, yps_grid, B_coarse, refine_factor=7):
    """
    For each coarse boundary cell, run a finer scan only inside that cell.
    Returns (xb, yb) midpoints of fine subcells that contain a boundary.
    """
    cells = boundary_cells(B_coarse)
    boundary_points = []

    for (iy, ix) in cells:
        x0, x1 = xps_grid[ix], xps_grid[ix+1]
        y0, y1 = yps_grid[iy], yps_grid[iy+1]

        xfine = np.linspace(x0, x1, refine_factor)
        yfine = np.linspace(y0, y1, refine_factor)

        Bf, _, _, _ = scan_bifurcation(xfine, yfine)

        # extract boundary points inside this refined cell
        for j in range(len(yfine)-1):
            for i in range(len(xfine)-1):
                blk = Bf[j:j+2, i:i+2]
                if np.min(blk) != np.max(blk):
                    boundary_points.append(((xfine[i] + xfine[i+1]) / 2,
                                            (yfine[j] + yfine[j+1]) / 2))
    return np.array(boundary_points)

if __name__ == "__main__":
    # Choose scan ranges (adjust to your biological regime)
    xps_values = np.linspace(20, 100, 50)  # 71 points
    yps_values = np.linspace(20, 100, 50)

    B, S, D, M = scan_bifurcation(xps_values, yps_values)
    
    bp = refine_cells(xps_values, yps_values, B, refine_factor=9)

    X0, Y0 = np.meshgrid(xps_values, yps_values)
    
    # ===== Export bistability map used by imshow =====
    # B has shape (len(yps_values), len(xps_values)) with B[iy, ix]
    # corresponding to (xps_values[ix], yps_values[iy])

    outname = "bistability_map_xps_yps-XY-PS.dat"
    with open(outname, "w") as f:
        f.write("# xps  yps  B\n")
        for iy, yps in enumerate(yps_values):
            for ix, xps in enumerate(xps_values):
                f.write(f"{xps:.6f} {yps:.6f} {int(B[iy, ix])}\n")
                f.write("\n")  # blank line between rows (optional)
        
    print("Saved:", outname)
    

    plt.figure()
    plt.contour(X0, Y0, B, levels=[0.5], linewidths=2)
    if bp.size > 0:
        plt.scatter(bp[:,0], bp[:,1], s=6)
    plt.xlabel("xps")
    plt.ylabel("yps")
    plt.title("Boundary: coarse contour + refined points")
    plt.show()


    # --- Plot bistability map ---
    plt.figure(figsize=(6.2, 5.2))
    plt.imshow(B, origin="lower",
               extent=[xps_values[0], xps_values[-1], yps_values[0], yps_values[-1]],
               aspect="auto")
    plt.xlabel("xps (X condensation threshold)")
    plt.ylabel("yps (Y condensation threshold)")
    plt.title("Bistability map (1=bistable, 0=monostable)")
    plt.colorbar(label="bistable")
    plt.tight_layout()

    # --- Plot boundary contour (Fig-2c analogue) ---
    X, Y = np.meshgrid(xps_values, yps_values)
    plt.figure(figsize=(6.2, 5.2))
    plt.contourf(X, Y, B, levels=[-0.5, 0.5, 1.5], alpha=0.6)
    cs = plt.contour(X, Y, B, levels=[0.5], linewidths=2.0)
    plt.xlabel("xps")
    plt.ylabel("yps")
    plt.title("Bistability boundary in (xps, yps)")
    plt.tight_layout()

    # Optional: show near-fold points by min|detJ| (useful as a consistency check)
    plt.figure(figsize=(6.2, 5.2))
    plt.contour(X, Y, M, levels=[DET_TOL], linewidths=2.0)
    plt.xlabel("xps")
    plt.ylabel("yps")
    plt.title(f"Near-fold indicator: contour of min|detJ| = {DET_TOL}")
    plt.tight_layout()

    plt.show()
    
    
   # ===================== Eigenvalues: export (REAL parts only) + labeled plots =====================

param_sets = [(50, 70), (50, 50), (70, 50)]

def fp_type_from_eigs(eig):
    r = np.real(eig)
    if np.all(r < 0):
        return "stable"
    if r[0] * r[1] < 0:
        return "saddle"
    return "unstable"

for (xps0, yps0) in param_sets:
    # --- compute fixed points + eigenvalues ---
    fps, eigs = spectrum_plot_at(xps0, yps0, guesses=INITIAL_GUESSES)

    # --- deduplicate fixed points more strictly (recommended) ---
    fps2, eigs2 = [], []
    for fp, eig in zip(fps, eigs):
        if not any(np.allclose(fp, g, atol=1e-4) for g in fps2):
            fps2.append(fp); eigs2.append(eig)
    fps, eigs = fps2, eigs2

    # medians for low/high labeling
    x_med = np.median([fp[0] for fp in fps])
    y_med = np.median([fp[1] for fp in fps])

    def fp_label_from_state(xp, yp):
        xl = "high" if xp >= x_med else "low"
        yl = "high" if yp >= y_med else "low"
        return f"{xl}X,{yl}Y"

    # --- export eigenvalues table (REAL parts only) ---
    rows = []
    for k, (fp, eig) in enumerate(zip(fps, eigs)):
        re = np.real(eig)
        fptype = fp_type_from_eigs(eig)
        fplabel = fp_label_from_state(fp[0], fp[1])
        rows.append([xps0, yps0, k, fp[0], fp[1], re[0], re[1]])

    out_dat = f"toggle_eigs_real_xps_{xps0}_yps_{yps0}-XY-PS.dat"
    np.savetxt(out_dat, np.array(rows, dtype=float),
               header="xps yps k xp yp Re(l1) Re(l2)",
               fmt="%.8g")
    print("Saved:", out_dat)

    # --- minimal 1D plot: Re(lambda) only, legend by fixed point label/type ---
    plt.figure(figsize=(6.0, 2.8))

    handles = {}
    for k, (fp, eig) in enumerate(zip(fps, eigs)):
        re = np.real(eig)
        fptype = fp_type_from_eigs(eig)
        fplabel = fp_label_from_state(fp[0], fp[1])

        marker = "o" if fptype == "stable" else ("s" if fptype == "saddle" else "x")

        h = plt.scatter([re[0], re[1]], [k, k], marker=marker)

        leg_text = f"{fplabel} ({fptype})"
        if leg_text not in handles:
            handles[leg_text] = h

    plt.axvline(0, linestyle="--")
    plt.yticks(range(len(fps)), [f"FP{k}" for k in range(len(fps))])
    plt.xlabel("Re(lambda)")
    plt.ylabel("fixed point index")
    plt.title(f"Jacobian real eigenvalues at fixed points (xps={xps0}, yps={yps0})")
    plt.legend(handles.values(), handles.keys(), frameon=False, loc="best")
    plt.tight_layout()
    plt.show()
    