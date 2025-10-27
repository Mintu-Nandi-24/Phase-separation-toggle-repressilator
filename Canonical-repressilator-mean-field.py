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



Canonical Repressilator: Mean-Field ODE Simulation
==================================================

This script simulates the canonical repressilator — a cyclic gene regulatory
network where three transcription factors (TFs) repress each other in sequence:
    X ⊣ Y ⊣ Z ⊣ X

The dynamics are modeled using a system of ordinary differential equations (ODEs)
in the mean-field (deterministic) regime.

Equations:
    dx/dt = α_x / (1 + (z/K_zx)^n) - x / τ_x
    dy/dt = α_y / (1 + (x/K_xy)^n) - y / τ_y
    dz/dt = α_z / (1 + (y/K_yz)^n) - z / τ_z

where:
    α_i   → synthesis rate of TF i
    τ_i   → lifetime (inverse degradation rate) of TF i
    K_ij  → dissociation constant for repression of j by i
    n     → cooperativity coefficient (Hill coefficient)

Outputs:
    • Time evolution of [X, Y, Z]
    • Saved CSV file of trajectories: "Canonical_repressilator_mf.csv"
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# Parameters
alpha_x, alpha_y, alpha_z = 5, 5, 5            #synthesis
tau_x, tau_y, tau_z = 1/0.02, 1/0.02, 1/0.02   #TF lifetime
bx, by, bz = 1/tau_x, 1/tau_y, 1/tau_z         # degradation
K_xy, K_zx, K_yz = 20, 20, 20                  #Dissociation constant






# Define the ODEs
def system(t, z, n):
    x, y, z = z
    dxdt = alpha_x * (1 / (1 + (z/K_zx)**n)) - bx * x
    dydt = alpha_y * (1 / (1 + (x/K_xy)**n)) - by * y
    dzdt = alpha_z * (1 / (1 + (y/K_yz)**n)) - bz * z
    return [dxdt, dydt, dzdt]

# Function to plot the evolution based on the parameter n
def plot_time_evolution(n):
    # Initial conditions

    x0 = 30 #0  # Initial value of X
    y0 = 38 #0  # Initial value of Y
    z0 = 31 #0  # Initial value of Z
    initial_conditions = [x0, y0, z0]

# Time span for the simulation
    t_span = (0, 3000)  # Start and end time
    t_eval = np.linspace(t_span[0], t_span[1], 5000)  # Time points to evaluate

# Solve the system of ODEs
    solution = solve_ivp(lambda t, z: system(t, z, n), t_span, initial_conditions, t_eval=t_eval)

# Extract the solution
    x_t = solution.y[0]
    y_t = solution.y[1]
    z_t = solution.y[2]
    t = solution.t

# Plotting the time evolution
    plt.figure(figsize=(15, 5))
    plt.plot(t, x_t, label='X', color='r')
    plt.plot(t, y_t, label='Y', color='b')
    plt.plot(t, z_t, label='Z', color='g')

# Labels and title
    plt.xlabel('Time')
    plt.ylabel('Copy Number')
    plt.title('Repressilator time trajectory')
    plt.legend()
    plt.show()

    return t, x_t, y_t, z_t

# Call the function to plot the evolution
t, x_t, y_t, z_t = plot_time_evolution(n=3) # put the value of cooperativity n here


from google.colab import files
np.savetxt("Cannonical_repressilator_mf.csv",
          np.column_stack((t, x_t, y_t, z_t)),
           delimiter=",",
           header="time,x,y,z",
           comments='')

files.download("Cannonical_repressilator_mf.csv")