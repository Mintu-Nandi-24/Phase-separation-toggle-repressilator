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
Toggle Switch Phase-Plane with Phase Separation in X (xp/xm) and Dilute Y
===============================================================================

Overview
--------
This script performs a deterministic phase-plane analysis of a toggle switch in
which transcription factor X can **phase-separate** into two subpopulations:
  • xp : dilute/soluble X (regulatory pool that represses Y)
  • xm : dense/condensed X (sequestered pool in a droplet/condensate)

Y remains non-phase-separating (single dilute pool). The model couples standard
Hill-type repression kinetics with a **diffusion-limited** dilute↔dense exchange
governed by a **free-energy** difference (detailed balance).

State variables (phase plane): (xp, y)
xm is treated **quasi-steadily** by solving dxm/dt = 0 for each xp.

Dynamics
--------
dxp/dt = a_x / (1 + (y/K_yx)^n) − b_x xp − τ * k_in(xp,xm) + τ * k_out(xp,xm)
dxm/dt = τ * k_in(xp,xm) − τ * k_out(xp,xm) − b_x xm
dy/dt  = a_y / (1 + (xp/K_xy)^n) − b_y y

• Hill repression acts through **xp** (the free/dilute X).
• k_in, k_out are dilute↔dense transfer rates:
      k_in(xp,xm)  = 0 (if xp < xps) else (6 D_c xp) / (V_tot − v xm)^(2/3)
      k_out(xp,xm) = k_in(xp+1,xm−1) * exp( [F(xp,xm) − F1(xp,xm)] / (k_B T) )

  with single-particle free energies:
      F (xp,xm)  = −C_P xm + k_B T · xp [ ln( xp / ((V_tot/v) − xm) ) − 1 ]
      F1(xp,xm) =  −C_P (xm−1) + k_B T · (xp+1) [ ln( (xp+1) / ( (V_tot/v) − (xm−1) ) ) − 1 ]

• Phase-separation threshold for X:
      xps = φ_s · (V_tot / v),  with  φ_s = xps / (V_tot/v)

Parameters & Symbols (units consistent with copy numbers / time)
----------------------------------------------------------------
a_x, a_y     : maximal synthesis rates of X, Y
b_x, b_y     : degradation rate constants
K_xy, K_yx   : dissociation constants (repression thresholds)
n            : Hill coefficient (cooperativity)
τ            : dimensionless scaling (exchange time prefactor)
v            : molecular volume of one TF
V_tot        : system volume
D_c          : effective diffusion coefficient ~ V_tot^(2/3) / (6 τ_D)
τ_D          : diffusion timescale (typically 0.1 * 1/b_x)
k_B T        : thermal energy
C_P          : −k_B T ( ln φ_s − φ_s )  (appears in free-energy)

What the script does
--------------------
1) **Nullclines**
   • X-nullcline (dXP/dt=0): computed by 1-D ODE projection with xm determined
     from dxm/dt=0 via root-finding (brentq).
   • Y-nullcline (dY/dt=0): computed similarly.

2) **Fixed points**
   • Multiple seeded solves (fsolve) for [dxp/dt, dy/dt]=0, with xm(xp) from
     the quasi-steady relation dxm/dt=0.
   • De-duplicated and filtered to lie near nullcline intersections.

3) **Stability**
   • Numerical Jacobian in (xp,y), eigenvalues used to classify:
     stable / unstable / saddle.

4) **Separatrix**
   • If a saddle exists, compute basin boundary by backward-time integration of
     the 2-D vector field (xp,y), nudged around the saddle (standard trick).

5) **Outputs**
   • Figure: nullclines + (optional) separatrix over (xp,y)
   • ASCII exports (two columns: X Y):
       - "x_nullcline-toggle-ps-y-30.dat"   (dX/dt = 0 curve)
       - "y_nullcline-toggle-ps-y-30.dat"   (dY/dt = 0 curve)
       - "separatrix-toggle-ps-y-30.dat"    (if a saddle is found)

Key functions (map to code)
---------------------------
F, F1        : free energies used for detailed-balance in k_out
kin, kout    : dilute↔dense transfer rates (thresholded by xps)
dxpdt, dxmdt : ODEs for dilute and dense X
dydt         : ODE for dilute Y
calculate_xm : solves dxmdt=0 for xm at given xp (brentq)
xp_nullcline_f, y_nullcline_f : compute nullclines by projecting ODEs
separatrix   : builds separatrix via backward integration near saddle

How to use / modify
-------------------
• Change **xps** (and hence φ_s) to set the X phase-separation threshold.
• Adjust kinetic (a, b, K, n) and transport (τ, τ_D, D_c) parameters as needed.
• Axis ranges: (xpi,xpf) and (yi,yf).
• Increase resolution or tolerances for smoother curves (may cost runtime).

Notes & caveats
---------------
• xm is treated quasi-steadily (dxm/dt=0); this is accurate when exchange is
  faster than degradation/production dynamics of xp and y.
• Numerical Jacobian is used because xm depends implicitly on xp via dxm/dt=0.
• If bracketing fails in brentq for some xp, xm is set to 0 as a fallback.

This deterministic analysis complements how phase separation reshapes the **basin geometry** and separatrix of
the toggle switch, when only X undergoes condensation.
===============================================================================
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import brentq

##################### Definitions ########################################
def F(xp, xm):
    return -CP * xm + kB * T * xp * (np.log(xp / ((Vtot/v) - xm)) - 1)

def F1(xp, xm):
    return -CP * (xm-1) + kB * T * (xp+1) * (np.log((xp+1) / (Vtot/v - (xm-1))) - 1)

def kin(xp, xm):
    if xp < xps:
        return 0
    else:
        return (6 * Dc * xp) / (Vtot - (v * xm))**(2.0/3.0)

def kout(xp, xm):
    if xp < xps:
        return 0
    else:
        return ((6 * Dc * (xp + 1)) / (Vtot - (v * (xm-1)))**(2.0/3.0)) * np.exp((F(xp, xm) - F1(xp, xm)) / (kB * T))

def dxpdt(xp, y, xm):
    return ax * (1 / (1 + (y/kyx)**n)) - bx * xp - t * kin(xp, xm) + t * kout(xp, xm)

def dxmdt(xp, xm):
    return t * kin(xp, xm) - t * kout(xp, xm) - bx * xm

def dydt(xp, y):
    return ay * (1 / (1 + (xp/kxy)**n)) - by * y

def vector_field(X):
    xp, y = X
    xm = calculate_xm(xp)
    return [dxpdt(xp, y, xm), dydt(xp, y)]

def xp_nullcline_f(y):
    def system(t, X):
        xp, xm = X
        return [dxpdt(xp, y, xm), 0]

    xp0 = 1
    xm0 = calculate_xm(xp0)

    sol = solve_ivp(system, [0, 100], [xp0, xm0], method='RK45', dense_output=True, rtol=1e-8, atol=1e-8)
    return sol.y[0, -1]

def y_nullcline_f(xp):
    def system(t, X):
        xp, y = X
        return [0, dydt(xp, y)]

    y0 = 1
    sol = solve_ivp(system, [0, 100], [xp, y0], method='RK45', dense_output=True, rtol=1e-8, atol=1e-8)
    return sol.y[1, -1]


def calculate_xm(xp):
    if xp < xps:
        return 0
    else:
        try:
            xm_root = brentq(lambda xm: dxmdt(xp, xm), 0, 500)
            return xm_root
        except ValueError:
            return 0  # fallback if no root in bracket

def separatrix(saddle_points, xpi, xpf, yi, yf):
    t_max = 90
    num_points = 100
    eps = 1e-2
    
    # Check if ther is no saddle point
    if not saddle_points:
        return
    
    # Negative time function to integrate to compute separatrix
    def rhs(ab,t):
        a, b = ab
        
        if xpi<a<xpf and yi<b<yf:
            return -np.array(vector_field(ab))
        else:
            return np.array([0,0])
    
    t = np.linspace(0, t_max, num_points)
    
    ab0_upper = np.array(saddle_points) + eps
    ab_upper = odeint(rhs, ab0_upper, t)
    
    ab0_lower = np.array(saddle_points) - eps
    ab_lower = odeint(rhs, ab0_lower, t)
    
    sep_a = np.concatenate((ab_lower[::-1,0], ab_upper[:,0]))
    sep_b = np.concatenate((ab_lower[::-1,1], ab_upper[:,1]))
    
    return sep_a, sep_b


############ Parameters #######################################################

# Parameters
ax = 5
ay = 5
bx = 0.05
by = 0.05
kxy = 30 #1
kyx = 30 #1
n = 2

t = 1

tau_x = 1 / bx

v = 1e-25
Vtot = 1e-20
V_factor = Vtot / v


xps = 30
phi_s = xps / V_factor

print("yp_s =",xps,",","Phis_s =", phi_s)

T = 305
R = 8.314
NA = 6.023e23
tau_D = 0.1 * tau_x

kB = R / NA
Dc = Vtot**(2 / 3) / (6 * tau_D)
CP = kB * T * (phi_s - np.log(phi_s))

xpi = 0 
xpf = 100 
yi = 0 
yf = 100 

xp_nullcline = np.linspace(xpi, xpf, 1000) 
y_nullcline = np.linspace(yi, yf, 1000)

########### Nullclines Plots ###################################################################

xp_nullcline_values = np.array([xp_nullcline_f(y) for y in y_nullcline])
y_nullcline_values = np.array([y_nullcline_f(xp) for xp in xp_nullcline])


plt.plot(y_nullcline_values, xp_nullcline, 'r-', label='dX/dt = 0', lw=1)
plt.plot(y_nullcline, xp_nullcline_values, 'b-', label=r"$dY_+/dt = 0$", lw=1)

########### Fixed points evaluation ################################################################################################

# Find fixed points
initial_guesses = [[0.1, 10], [10, 0.1], [1, 1], [10, 10], [20, 20], [50, 50], [60, 60], [70, 70], [100, 100]]
fixed_points = []

for guess in initial_guesses:
    fixed_point = fsolve(lambda X: [dxpdt(X[0], X[1], calculate_xm(X[0])), dydt(X[0], X[1])], guess, xtol=1e-10)
    if not any(np.allclose(fixed_point, fp, atol=1e-3) for fp in fixed_points):  # Avoid duplicates
        fixed_points.append(fixed_point)

# Only keep fixed points that are close to the nullcline intersection
nullcline_threshold = 1e-0  # Adjust based on the system's scale

valid_fixed_points = []
# for fp in valid_fixed_points_1:
for fp in fixed_points:
    xp_null = xp_nullcline_f(fp[1])  # xp value on the nullcline
    y_null = y_nullcline_f(fp[0])  # y value on the nullcline
    # print(abs(fp[0] - xp_null), abs(fp[1] - y_null))
    
    
    if abs(fp[0] - xp_null) < nullcline_threshold and abs(fp[1] - y_null) < nullcline_threshold:
        valid_fixed_points.append(fp)
        

# Stability Analysis
def jacobian(xp, y):
    eps = 1e-6
    dF_dxp = (dxpdt(xp + eps, y, calculate_xm(xp + eps)) - dxpdt(xp, y, calculate_xm(xp))) / eps
    dF_dy = (dxpdt(xp, y + eps, calculate_xm(xp)) - dxpdt(xp, y, calculate_xm(xp))) / eps
    dG_dxp = (dydt(xp + eps, y) - dydt(xp, y)) / eps
    dG_dy = (dydt(xp, y + eps) - dydt(xp, y)) / eps
    return np.array([[dF_dxp, dF_dy], [dG_dxp, dG_dy]])


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
    sep_a, sep_b = separatrix(saddle_points, xpi, xpf, yi, yf)
    plt.plot(sep_b, sep_a, 'g-', lw=1)#, label='Separatrix')


plt.legend(frameon=False)
plt.xlim(xpi, xpf)
plt.ylim(yi, yf)
plt.xlabel("Protein X", fontsize=14)
plt.ylabel("Protein Y in dilute phase", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14, frameon=False)

# plt.savefig("toggle-phaseY-ps-3e_4-yps-30.pdf", format="pdf", bbox_inches="tight")

plt.show()



# === Export nullclines as .dat files ===
with open("x_nullcline-toggle-ps-y-30.dat", "w") as fx:
    for x, y in zip(y_nullcline_values.flatten(), xp_nullcline):
        fx.write(f"{x:.6f} {y:.6f}\n")

with open("y_nullcline-toggle-ps-y-30.dat", "w") as fy:
    for x, y in zip(y_nullcline, xp_nullcline_values.flatten()):
        fy.write(f"{x:.6f} {y:.6f}\n")

# === Export separatrix as .dat file ===
if sep_a is not None and sep_b is not None:
    with open("separatrix-toggle-ps-y-30.dat", "w") as fsep:
        for x, y in zip(sep_a, sep_b):
            fsep.write(f"{x:.6f} {y:.6f}\n")

# Print fixed points
for i, fp in enumerate(valid_fixed_points):
    xm_value = calculate_xm(fp[0])
    print(f"Fixed Point {i+1}: xp = {fp[0]}, y = {fp[1]}, xm = {xm_value}")
    print(f"dx/dt at fixed point: {dxpdt(fp[0], fp[1], xm_value)}")
    print(f"dy/dt at fixed point: {dydt(fp[0], fp[1])}")
    print(f"Eigenvalue: {eigens[i]}\n")
    
