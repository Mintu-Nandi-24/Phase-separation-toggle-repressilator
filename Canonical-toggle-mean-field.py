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
Deterministic Toggle Switch Phase-Plane Analysis
===============================================================================
-------------------------------------------------------------------------------
Overview
-------------------------------------------------------------------------------
This script performs a **deterministic phase-plane analysis** of a canonical
**genetic toggle switch**, a minimal bistable motif formed by two transcription
factors (TFs), X and Y, that mutually repress each other:

    X ⊣ Y  and  Y ⊣ X

Each TF inhibits the synthesis of the other through a Hill-type repression
function. The system is governed by two coupled ODEs:

    dX/dt = αₓ / (1 + (Y/K_yx)ⁿ) − βₓ X
    dY/dt = α_y / (1 + (X/K_xy)ⁿ) − β_y Y

where:
    αᵢ : maximal synthesis rate of TF i
    βᵢ : degradation rate constant of TF i
    K_ij : dissociation constant for repression
    n : Hill coefficient (cooperativity)

-------------------------------------------------------------------------------
Objectives
-------------------------------------------------------------------------------
1. Compute and plot the **nullclines** (dX/dt=0, dY/dt=0)
2. Identify all **fixed points** (intersections of nullclines)
3. Perform **stability analysis** by evaluating eigenvalues of the Jacobian
4. Locate **saddle points** and integrate their **separatrix**, which divides
   the two basins of attraction (X-dominant and Y-dominant states)
5. Export all computed curves as ASCII `.dat` files for downstream plotting

-------------------------------------------------------------------------------
Exported Data Files
-------------------------------------------------------------------------------
    • x_nullcline-toggle.dat
        (X-nullcline: points where dX/dt = 0)

    • y_nullcline-toggle.dat
        (Y-nullcline: points where dY/dt = 0)

    • separatrix-toggle.dat
        (Separatrix dividing the two stable basins)

Each file contains two columns: [X, Y]

-------------------------------------------------------------------------------
Usage
-------------------------------------------------------------------------------
Simply run this script. It will:
    - Generate the phase-plane plot (nullclines, fixed points, separatrix)
    - Print the coordinates and eigenvalues of all fixed points
    - Export the .dat files to the working directory

-------------------------------------------------------------------------------
Notes
-------------------------------------------------------------------------------
• The separatrix is computed by integrating backward along the stable
  eigenvector of the saddle point (negative eigenvalue direction).
• Parameters (α, β, K, n) can be easily modified for parametric studies.
• This deterministic framework complements stochastic Gillespie simulations
  to visualize how noise drives switching across the deterministic barrier.

===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
#from scipy.integrate import solve_ivp
from scipy.integrate import odeint


def dxdt(x, y):
    return ax * (1 / (1 + (y/kyx)**n)) - bx * x

def dydt(x, y):
    return ay * (1 / (1 + (x/kxy)**n)) - by * y

def vector_field(X):
    x, y = X
    return [dxdt(x, y), dydt(x, y)]

def x_nullcline_f(y):
    return fsolve(lambda x: dxdt(x, y), 1)

def y_nullcline_f(x):
    return fsolve(lambda y: dydt(x, y), 1)

def separatrix(saddle_points, xi, xf, yi, yf):
    t_max = 100
    num_points = 100
    eps = 1e-2
    
    # Check if ther is no saddle point
    if not saddle_points:
        return
    
    # Negative time function to integrate to compute separatrix
    def rhs(ab,t):
        a, b = ab
        
        if xi<a<xf and yi<b<yf:
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
    

   
# Parameters
ax=5 #5
ay=5 #5
bx=0.05 #0.05
by=0.05 #0.05
kxy = 30
kyx = 30
n=2

xi = 0
xf = 100 #150
yi = 0
yf = 100 #150

xr = np.linspace(xi, xf, 30)
yr = np.linspace(yi, yf, 30)

x_nullcline = np.linspace(xi, xf, 100)
y_nullcline = np.linspace(yi, yf, 100)

x_nullcline_values = np.array([x_nullcline_f(y) for y in y_nullcline])
y_nullcline_values = np.array([y_nullcline_f(x) for x in x_nullcline])

plt.plot(x_nullcline_values, y_nullcline, 'r-', label='dX/dt = 0', linewidth=1)
plt.plot(x_nullcline, y_nullcline_values, 'b-', label='dY/dt = 0', linewidth=1)


# Use multiple initial guesses for fsolve
initial_guesses = [[10,100], [100,10], [100, 100]]#, [20, 20],[50,50],[100,100],[200,200]]
fixed_points = []

for guess in initial_guesses:
    fixed_point = fsolve(lambda X: [dxdt(X[0], X[1]), dydt(X[0], X[1])], guess)
    #print(fixed_point)
    if not any(np.allclose(fixed_point, fp, atol=1e-3) for fp in fixed_points):  # Avoid duplicates
        fixed_points.append(fixed_point)


# Define a threshold for dnpdt and dnmdt to be considered as fixed points
threshold = 1e-3
valid_fixed_points = []

for fp in fixed_points:
    dxdt_val = dxdt(fp[0], fp[1])
    dydt_val = dydt(fp[0], fp[1])
    if np.abs(dxdt_val) < threshold and np.abs(dydt_val) < threshold:
        valid_fixed_points.append(fp)
        # plt.plot(fp[0], fp[1], 'mo', markersize=10)

# Stability Analysis
def jacobian(x, y):
    eps = 1e-6
    dF_dx = (dxdt(x + eps, y) - dxdt(x, y)) / eps
    dF_dy = (dxdt(x, y + eps) - dxdt(x, y)) / eps
    dG_dx = (dydt(x + eps, y) - dydt(x, y)) / eps
    dG_dy = (dydt(x, y + eps) - dydt(x, y)) / eps
    return np.array([[dF_dx, dF_dy], [dG_dx, dG_dy]])

stable_label = True
unstable_label = True
saddle_label = True
eigens = []
saddle_points = []
for fp in valid_fixed_points:
    J = jacobian(fp[0], fp[1])
    eigenvalues = np.linalg.eigvals(J)
    eigens.append(eigenvalues)
    
    # Annotate fixed points with their coordinates
    #plt.annotate(f"{fp[0]:.2f}, {fp[1]:.2f}", (fp[0], fp[1]), textcoords="offset points", xytext=(5,5), ha='left')
    
    if np.all(eigenvalues < 0):
        # if stable_label:
        #     plt.plot(fp[0], fp[1], 'go', markersize=10, label='Stable')
        #     stable_label = False
        # else:
        #     plt.plot(fp[0], fp[1], 'go', markersize=10)
        continue
    elif np.all(eigenvalues > 0):
        # if unstable_label:
        #     plt.plot(fp[0], fp[1], 'ro', markersize=10, label='Unstable')
        #     unstable_label = False
        # else:
        #     plt.plot(fp[0], fp[1], 'ro', markersize=10)
        continue
    else:
        # if saddle_label:
        #     plt.plot(fp[0], fp[1], 'yo', markersize=10, label='Saddle')
        #     saddle_label = False
        # else:
        #     plt.plot(fp[0], fp[1], 'yo', markersize=10)
        saddle_points.append(fp)
        
    

saddle_points = saddle_points[0].tolist()

# print(saddle_points)

sep_a, sep_b = separatrix(saddle_points, xi, xf, yi, yf)
plt.plot(sep_a, sep_b, 'g-', linewidth=1)#, label='Separatrix')


plt.legend(frameon=False)
plt.xlim(xi, xf)
plt.ylim(yi, yf)
#plt.title("Toggle Switch")
# plt.xscale("log")
# plt.yscale("log")
plt.xlabel("Protein X", fontsize=14)
plt.ylabel("Protein Y", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14, frameon=False)


# plt.savefig("simple-toggle.pdf", format="pdf", bbox_inches="tight")

plt.show()

# === Export nullclines as .dat files ===
with open("x_nullcline-toggle.dat", "w") as fx:
    for x, y in zip(x_nullcline_values.flatten(), y_nullcline):
        fx.write(f"{x:.6f} {y:.6f}\n")

with open("y_nullcline-toggle.dat", "w") as fy:
    for x, y in zip(x_nullcline, y_nullcline_values.flatten()):
        fy.write(f"{x:.6f} {y:.6f}\n")

# === Export separatrix as .dat file ===
if sep_a is not None and sep_b is not None:
    with open("separatrix-toggle.dat", "w") as fsep:
        for x, y in zip(sep_a, sep_b):
            fsep.write(f"{x:.6f} {y:.6f}\n")


#print(f"Fixed points: {fixed_points}")

# Validate the found fixed points
for i, fp in enumerate(fixed_points):
    print(f"Fixed Point {i+1}: x = {fp[0]}, y = {fp[1]}")
    print(f"dx/dt at fixed point: {dxdt(fp[0], fp[1])}")
    print(f"dy/dt at fixed point: {dydt(fp[0], fp[1])}")
    print(f"Eigenvalue: {eigens[i]}\n")
