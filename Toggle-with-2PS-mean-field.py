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



===============================================================================
Deterministic Toggle Switch with Phase Separation in Both X and Y
===============================================================================
-------------------------------------------------------------------------------
Overview
-------------------------------------------------------------------------------
This script performs a **deterministic phase-plane analysis** of a genetic
**toggle switch** where *both* transcription factors (TFs) X and Y can
**undergo phase separation** into dilute and dense pools.

The system models four molecular populations:
    • xp — X in dilute (soluble) phase
    • xm — X in dense (condensed) phase
    • yp — Y in dilute (soluble) phase
    • ym — Y in dense (condensed) phase

Each TF represses the other's synthesis via a Hill-type function that depends
on the dilute copy number (xp and yp). Phase separation introduces additional
kinetic terms describing diffusion-limited exchange between dilute and dense
phases governed by thermodynamic free-energy differences.

-------------------------------------------------------------------------------
Model Equations
-------------------------------------------------------------------------------
The coupled ODEs are:

    dxp/dt = a_x / (1 + (y_p/K_yx)^n) − b_x xp − τ k_in^x(xp,xm) + τ k_out^x(xp,xm)
    dxm/dt = τ k_in^x(xp,xm) − τ k_out^x(xp,xm) − b_x xm
    dyp/dt = a_y / (1 + (x_p/K_xy)^n) − b_y yp − τ k_in^y(yp,ym) + τ k_out^y(yp,ym)
    dym/dt = τ k_in^y(yp,ym) − τ k_out^y(yp,ym) − b_y ym

Here:
    - k_in, k_out  : diffusion-limited exchange rates (dilute ↔ dense)
    - τ            : characteristic timescale scaling factor
    - Free-energy differences control k_out via detailed balance

Each TF i ∈ {x,y} follows:
    k_in^i(p,m)  = 0 if p < p*_i else (6 D_i p) / (V_tot − v m)^(2/3)
    k_out^i(p,m) = k_in^i(p+1,m−1) * exp( [F_i(p,m) − F1_i(p,m)] / (k_B T) )

Free energy functions (for i = x or y):
    F_i(p,m)  = −C_P^i m + k_B T · p [ ln(p / ((V_tot/v) − m)) − 1 ]
    F1_i(p,m) = −C_P^i (m−1) + k_B T · (p+1) [ ln((p+1)/((V_tot/v) − (m−1))) − 1 ]

Thresholds for phase separation:
    p*_x = φ_x^* (V_tot / v)
    p*_y = φ_y^* (V_tot / v)

-------------------------------------------------------------------------------
Assumptions
-------------------------------------------------------------------------------
• xm and ym are treated as quasi-steady-state variables:
      dxm/dt = 0,  dym/dt = 0
  solved numerically for xm(xp) and ym(yp) using brentq root-finding.

• Phase separation only occurs when xp ≥ xps or yp ≥ yps.
• Exchange dynamics are diffusion-limited, using D_cx, D_cy ∝ V_tot^(2/3)/(6 τ_D).

-------------------------------------------------------------------------------
Workflow
-------------------------------------------------------------------------------
1. **Define free energies and kinetic rates** for both X and Y.
2. **Compute nullclines**:
      - X-nullcline (dX_+/dt = 0)
      - Y-nullcline (dY_+/dt = 0)
3. **Find fixed points** via multi-start root solving (fsolve) on [dxp/dt, dyp/dt].
4. **Classify fixed points** by computing eigenvalues of the Jacobian matrix.
5. **Compute separatrix** (basin boundary) by integrating backward from the saddle.
6. **Plot phase plane** with nullclines, fixed points, and separatrix.
7. **Export curves** as ASCII `.dat` files for use in plotting software.

-------------------------------------------------------------------------------
Exported Files
-------------------------------------------------------------------------------
    x_nullcline-toggle-ps-x-70-y-50.dat   →  dX_+/dt = 0 nullcline
    y_nullcline-toggle-ps-x-70-y-50.dat   →  dY_+/dt = 0 nullcline
    separatrix-toggle-ps-x-70-y-50.dat    →  separatrix (if saddle exists)

All files contain two columns: [X_dilute, Y_dilute].

-------------------------------------------------------------------------------
Key Parameters
-------------------------------------------------------------------------------
a_x, a_y  : synthesis rates
b_x, b_y  : degradation rates
K_xy, K_yx : repression thresholds (dissociation constants)
n         : Hill coefficient
τ         : scaling factor for exchange kinetics
v         : molecular volume of one TF molecule
V_tot     : system volume
xps, yps  : phase-separation thresholds (copy numbers)
D_cx, D_cy : effective diffusion coefficients
C_Px, C_Py : free-energy prefactors from φ_s values
k_B, T    : thermal energy constants

-------------------------------------------------------------------------------
Interpretation
-------------------------------------------------------------------------------
This model generalizes the standard bistable toggle switch by incorporating
molecular condensation of both transcription factors. It reveals how phase
separation modifies the basin of attraction and the separatrix between the
two stable gene-expression states.

When both X and Y phase-separate, the topology of the nullclines and separatrix
depends critically on the relative condensation thresholds (xps, yps). The model
thus connects *biophysical phase behavior* with *dynamical stability* in gene
regulatory motifs.

===============================================================================
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import brentq

##################### Definitions ########################################
# For X


def Fx(xp, xm):
    return -CPx * xm + kB * T * xp * (np.log(xp / ((Vtot/v) - xm)) - 1)


def F1x(xp, xm):
    return -CPx * (xm-1) + kB * T * (xp+1) * (np.log((xp+1) / (Vtot/v - (xm-1))) - 1)


def kinx(xp, xm):
    if xp < xps:
        return 0
    else:
        return (6 * Dcx * xp) / (Vtot - (v * xm))**(2.0/3.0)


def koutx(xp, xm):
    if xp < xps:
        return 0
    else:
        return ((6 * Dcx * (xp + 1)) / (Vtot - (v * (xm-1)))**(2.0/3.0)) * np.exp((Fx(xp, xm) - F1x(xp, xm)) / (kB * T))


# For Y
def Fy(yp, ym):
    return -CPy * ym + kB * T * yp * (np.log(yp / ((Vtot/v) - ym)) - 1)


def F1y(yp, ym):
    return -CPy * (ym-1) + kB * T * (yp+1) * (np.log((yp+1) / (Vtot/v - (ym-1))) - 1)


def kiny(yp, ym):
    if yp < yps:
        return 0
    else:
        return (6 * Dcy * yp) / (Vtot - (v * ym))**(2.0/3.0)


def kouty(yp, ym):
    if yp < yps:
        return 0
    else:
        return ((6 * Dcy * (yp + 1)) / (Vtot - (v * (ym-1)))**(2.0/3.0)) * np.exp((Fy(yp, ym) - F1y(yp, ym)) / (kB * T))



# ODEs

def dxpdt(xp, yp, xm):
    return ax * (1 / (1 + (yp/kyx)**n)) - bx * xp - t * kinx(xp, xm) + t * koutx(xp, xm)


def dxmdt(xp, xm):
    return t * kinx(xp, xm) - t * koutx(xp, xm) - bx * xm


def dypdt(xp, yp, ym):
    return ay * (1 / (1 + (xp/kxy)**n)) - by * yp - t * kiny(yp, ym) + t * kouty(yp, ym)

def dymdt(yp, ym):
    return t * kiny(yp, ym) - t * kouty(yp, ym) - by * ym


def vector_field(X):
    xp, yp = X
    xm = calculate_xm(xp)
    ym = calculate_xm(yp)
    return [dxpdt(xp, yp, xm), dypdt(xp, yp, ym)]


def xp_nullcline_f(yp):
    def system(t, X):
        xp, xm = X
        return [dxpdt(xp, yp, xm), 0]

    xp0 = 1
    xm0 = calculate_xm(xp0)

    sol = solve_ivp(system, [0, 100], [xp0, xm0], method='RK45',
                    dense_output=True, rtol=1e-8, atol=1e-8)
    return sol.y[0, -1]


def yp_nullcline_f(xp):
    def system(t, X):
        yp, ym = X
        return [dypdt(xp, yp, ym), 0]

    yp0 = 1
    ym0 = calculate_ym(yp0)
    
    sol = solve_ivp(system, [0, 100], [yp0, ym0], method='RK45',
                    dense_output=True, rtol=1e-8, atol=1e-8)
    return sol.y[0, -1]

def calculate_xm(xp):
    if xp < xps:
        return 0
    else:
        try:
            xm_root = brentq(lambda xm: dxmdt(xp, xm), 0, 500)
            return xm_root
        except ValueError:
            return 0  # fallback if no root in bracket

# def calculate_xm(xp):
#     if xp < xps:
#         return 0
#     else:
#         xm_solution = fsolve(lambda xm: dxmdt(xp, xm), 1, xtol=1e-8)[0]
#         return max(0, xm_solution)

def calculate_ym(yp):
    if yp < yps:
        return 0
    else:
        try:
            ym_root = brentq(lambda ym: dymdt(yp, ym), 0, 500)
            return ym_root
        except ValueError:
            return 0  # fallback if no root in bracket
    
# def calculate_ym(yp):
#     if yp < yps:
#         return 0
#     else:
#         ym_solution = fsolve(lambda ym: dymdt(yp, ym), 1, xtol=1e-8)[0]
#         return max(0, ym_solution)


def separatrix(saddle_points, xpi, xpf, ypi, ypf):
    t_max = 99
    num_points = 100
    eps = 1e-2

    # Check if ther is no saddle point
    if not saddle_points:
        return

    # Negative time function to integrate to compute separatrix
    def rhs(ab, t):
        a, b = ab

        if xpi < a < xpf and ypi < b < ypf:
            return -np.array(vector_field(ab))
        else:
            return np.array([0, 0])

    t = np.linspace(0, t_max, num_points)

    ab0_upper = np.array(saddle_points) + eps
    ab_upper = odeint(rhs, ab0_upper, t)

    ab0_lower = np.array(saddle_points) - eps
    ab_lower = odeint(rhs, ab0_lower, t)

    sep_a = np.concatenate((ab_lower[::-1, 0], ab_upper[:, 0]))
    sep_b = np.concatenate((ab_lower[::-1, 1], ab_upper[:, 1]))

    return sep_a, sep_b


############ Parameters #######################################################

# Parameters
ax = 5
ay = 5
bx = 0.05
by = 0.05
kxy = 30  # 1
kyx = 30  # 1
n = 2

t = 1

tau_x = 1 / bx
tau_y = 1 / by

v = 1e-25
Vtot = 1e-20
V_factor = Vtot / v


xps = 60
xm_c = xps
phix_s = xps / V_factor

yps = 50
ym_c = yps
phiy_s = yps / V_factor


print("xp_s =", xps, ",", "Phix_s =", phix_s)
print("yp_s =", yps, ",", "Phiy_s =", phiy_s)


T = 305
R = 8.314
NA = 6.023e23
tau_Dx = 0.1 * tau_x
tau_Dy = 0.1 * tau_y

kB = R / NA
Dcx = Vtot**(2 / 3) / (6 * tau_Dx)
Dcy = Vtot**(2 / 3) / (6 * tau_Dy)
CPx = kB * T * (phix_s - np.log(phix_s))
CPy = kB * T * (phiy_s - np.log(phiy_s))

xpi = 0  # 0.001
xpf = 100  # 1000
ypi = 0  # 0.001
ypf = 100  # 1000

xp_nullcline = np.linspace(xpi, xpf, 1000)  # 10000)
yp_nullcline = np.linspace(ypi, ypf, 1000)  # 10000)
########### Nullclines Plots ###################################################################

xp_nullcline_values = np.array([xp_nullcline_f(yp) for yp in yp_nullcline])
yp_nullcline_values = np.array([yp_nullcline_f(xp) for xp in xp_nullcline])

plt.plot(xp_nullcline_values, yp_nullcline, 'r-', label=r"$dX_+/dt = 0$", lw=1)
plt.plot(xp_nullcline, yp_nullcline_values, 'b-', label=r"$dY_+/dt = 0$", lw=1)

########### Fixed points evaluation ################################################################################################

# Find fixed points
initial_guesses = [[0.1, 10], [10, 0.1], [1, 1], [
    10, 10], [20, 20], [20, 70], [50, 50], [70, 70], [70, 20], [100, 100]]
fixed_points = []

for guess in initial_guesses:
    fixed_point = fsolve(lambda X: [dxpdt(X[0], X[1], calculate_xm(
        X[0])), dypdt(X[0], X[1], calculate_ym(
            X[1]))], guess, xtol=1e-10)
    if not any(np.allclose(fixed_point, fp, atol=1e-3) for fp in fixed_points):  # Avoid duplicates
        fixed_points.append(fixed_point)

# Only keep fixed points that are close to the nullcline intersection
nullcline_threshold = 1e0  # Adjust based on the system's scale

valid_fixed_points = []
for fp in fixed_points:
    xp_null = xp_nullcline_f(fp[1])  # xp value on the nullcline
    yp_null = yp_nullcline_f(fp[0])  # y value on the nullcline
    # print(abs(fp[0] - xp_null), abs(fp[1] - yp_null))

    if abs(fp[0] - xp_null) < nullcline_threshold and abs(fp[1] - yp_null) < nullcline_threshold:
        valid_fixed_points.append(fp)

def jacobian(xp, yp):
    eps = 1e-6
    dF_dxp = (dxpdt(xp + eps, yp, calculate_xm(xp + eps)) - dxpdt(xp, yp, calculate_xm(xp))) / eps
    dF_dyp = (dxpdt(xp, yp + eps, calculate_xm(xp)) - dxpdt(xp, yp, calculate_xm(xp))) / eps
    dG_dxp = (dypdt(xp + eps, yp, calculate_ym(yp)) - dypdt(xp, yp, calculate_ym(yp))) / eps
    dG_dyp = (dypdt(xp, yp + eps, calculate_ym(yp+eps)) - dypdt(xp, yp, calculate_ym(yp))) / eps
    return np.array([[dF_dxp, dF_dyp], [dG_dxp, dG_dyp]])



eigens = []
saddle_points = []
for fp in valid_fixed_points:
    J = jacobian(fp[0], fp[1])
    eigenvalues = np.linalg.eigvals(J)
    eigens.append(eigenvalues)

    if np.all(eigenvalues < 0):
        continue
    elif np.all(eigenvalues > 0):
        continue
    else:
        saddle_points.append(fp)



if len(saddle_points) > 0:
    saddle_points = saddle_points[0].tolist()
    sep_a, sep_b = separatrix(saddle_points, xpi, xpf, ypi, ypf)
    plt.plot(sep_a, sep_b, 'g-', lw=1) #, label='Separatrix')


plt.legend(frameon=False)
plt.xlim(xpi, xpf)
plt.ylim(ypi, ypf)

plt.xlabel("Protein X in dilute phase", fontsize=14)
plt.ylabel("Protein Y in dilute phase", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14, frameon=False)

# plt.savefig("toggle-phaseXY-ps-5p1e_4-5e_4-xps-51-yps-50.pdf", format="pdf", bbox_inches="tight")

plt.show()

# === Export nullclines as .dat files ===
with open("x_nullcline-toggle-ps-x-70-y-50.dat", "w") as fx:
    for x, y in zip(xp_nullcline_values.flatten(), yp_nullcline):
        fx.write(f"{x:.6f} {y:.6f}\n")

with open("y_nullcline-toggle-ps-x-70-y-50.dat", "w") as fy:
    for x, y in zip(xp_nullcline, yp_nullcline_values.flatten()):
        fy.write(f"{x:.6f} {y:.6f}\n")

# === Export separatrix as .dat file ===
if sep_a is not None and sep_b is not None:
    with open("separatrix-toggle-ps-x-70-y-50.dat", "w") as fsep:
        for x, y in zip(sep_a, sep_b):
            fsep.write(f"{x:.6f} {y:.6f}\n")
            

# Print fixed points
for i, fp in enumerate(valid_fixed_points):
    xm_value = calculate_xm(fp[0])
    ym_value = calculate_ym(fp[1])
    print(f"Fixed Point {i+1}: xp = {fp[0]}, yp = {fp[1]}, xm = {xm_value}, ym = {ym_value}")
    print(f"dx/dt at fixed point: {dxpdt(fp[0], fp[1], xm_value)}")
    print(f"dy/dt at fixed point: {dypdt(fp[0], fp[1], ym_value)}")
    print(f"Eigenvalue: {eigens[i]}\n")
