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



Repressilator with Phase Separation in All Transcription Factors (Mean Field)
=============================================================================

This script simulates a repressilator — a cyclic gene regulatory motif
where three transcription factors (TFs) repress one another:

    X ⊣ Y ⊣ Z ⊣ X

Here, **all three TFs (X, Y, Z)** undergo *phase separation* into:
    - Dilute (cytoplasmic) phase: x_dilt, y_dilt, z_dilt
    - Dense (condensed/droplet) phase: x_dens, y_dens, z_dens

Regulatory interactions act through the **dilute-phase TFs**, while
the dense phase exchanges molecules via diffusion-limited transfer
governed by free-energy differences between the phases.

Outputs:
    • Time evolution of x_dilt, y_dilt, z_dilt
    • CSV file: "repressilator_phase_x_80_y_80_z_40.csv"
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters
alpha_x, alpha_y, alpha_z = 5, 5, 5                             #synthesis rate constant
tau_x, tau_y, tau_z = 1/0.02, 1/0.02, 1/0.02                    #TF lifetime
b_x, b_y, b_z = 1/tau_x, 1/tau_y, 1/tau_z                       #Degradation rate constant
n = 3                                                           #Cooperativity
K_xy, K_yz, K_zx = 20, 20, 20                                   # Dissociation constant

# Phase separation constants
v = 1.0e-25                                                    #Volume of one molecule
volume_factor = 1e5
Vtot = volume_factor * v                                       #Total volume
phi_star_x = 0.0008
phi_star_y = 0.0008
phi_star_z = 0.0004
x_star = phi_star_x * volume_factor                           #phase separation threshold for x
y_star = phi_star_y*volume_factor                             #phase separation threshold for y
z_star = phi_star_z*volume_factor                             #phase separation threshold for z
thermal_energy = 1.38e-23 * 305                               #thermal energy K_B *T
mu_x = -thermal_energy * (np.log(phi_star_x) - phi_star_x)    #Chemical potential
mu_y = -thermal_energy * (np.log(phi_star_y) - phi_star_y)    #Chemical potential
mu_z = -thermal_energy * (np.log(phi_star_z) - phi_star_z)    #Chemical potential
tau_d = 0.1 * tau_z                                           #Diffusion time
D = Vtot**(2. / 3.) / (6. * tau_d)                            #Diffusion coefficient

# Free energy function
def free_energy_x(dilt, bar):                                                 #calculation of free energy function in x
    eps = 1e-12
    denom = max(volume_factor - (bar - dilt), eps)
    temp1 = dilt / denom
    temp1 = max(temp1, eps)
    term1 = -mu_x * (bar - dilt)
    term2 = thermal_energy * dilt * (np.log(temp1) - 1)
    return term1 + term2

# Free energy function
def free_energy_y(dilt, bar):                                                #calculation of free energy function in y
    eps = 1e-12
    denom = max(volume_factor - (bar - dilt), eps)
    temp1 = dilt / denom
    temp1 = max(temp1, eps)
    term1 = -mu_y * (bar - dilt)
    term2 = thermal_energy * dilt * (np.log(temp1) - 1)
    return term1 + term2

# Free energy function
def free_energy_z(dilt, bar):                                                #calculation of free energy function in z
    eps = 1e-12
    denom = max(volume_factor - (bar - dilt), eps)
    temp1 = dilt / denom
    temp1 = max(temp1, eps)
    term1 = -mu_z * (bar - dilt)
    term2 = thermal_energy * dilt * (np.log(temp1) - 1)
    return term1 + term2

# Repressilator with phase separation in x, y and z
def repressilator_phase_sep(t, y):
    x_dilt, x_dens, y_dilt, y_dens, z_dilt, z_dens = y
    x_dilt, x_dens, y_dilt, y_dens, z_dilt, z_dens = map(lambda v: max(v, 0), (x_dilt,x_dens, y_dilt, y_dens, z_dilt, z_dens))

    # Total TF levels
    x_bar = x_dilt + x_dens                                                              #total TF levels in x
    y_bar = y_dilt + y_dens                                                              #total TF levels in y
    z_bar = z_dilt + z_dens                                                              #total TF levels in z

    # Condition 1: Before either phase separates
    if x_dilt < x_star and x_dens ==0 and y_dilt < y_star and y_dens == 0 and z_dilt < z_star and z_dens == 0:
        dxdilt_dt = alpha_x / (1 + (z_dilt / K_zx)**n) - b_x * x_dilt                      #x_dilt dynamics before phase separation in x, no exchange of molecule to droplet phase (no terms like k_in_x,k_outx)
        dydilt_dt = alpha_y / (1 + (x_dilt / K_xy)**n) - b_y * y_dilt                      #y_dilt dynamics before phase separation in y, no exchange of molecule to droplet phase (no terms like k_in_y,k_out_y)
        dzdilt_dt = alpha_z / (1 + (y_dilt / K_yz)**n) - b_z * z_dilt                      #z_dilt dynamics before phase separation in z, no exchange of molecule to droplet phase (no terms like k_in_z,k_out_z)
        dxdens_dt = 0                                                                      #no x droplet formation before phase separation in x
        dydens_dt = 0                                                                      #no y droplet formation before phase separation in y
        dzdens_dt = 0                                                                      #no z droplet formation before phase separation in z

    else:

        # ---- Phase separation in x ----
        if x_dilt >= x_star or x_dens > 0:
            deltaF_x = free_energy_x(x_dilt + 1, x_bar) - free_energy_x(x_dilt, x_bar)           #change in free energy for x
            k_in_x = (6 * D * x_dilt) / ((Vtot - v * x_dens)**(2./3.))                           #Dilute to droplet transfer in x
            k_out_x = k_in_x * np.exp(-deltaF_x / thermal_energy) if x_dens > 0 else 0.0         #Droplet to dilute tranfer in x
        else:
            k_in_x, k_out_x = 0, 0                                                               #no dilute to droplet transfer in x if the threshold is not exceeded


        # ---- Phase separation in y ----
        if y_dilt >= y_star or y_dens > 0:
            deltaF_y = free_energy_y(y_dilt + 1, y_bar) - free_energy_y(y_dilt, y_bar)           #change in free energy for y
            k_in_y = (6 * D * y_dilt) / ((Vtot - v * y_dens)**(2./3.))                           #Dilute to droplet transfer in y
            k_out_y = k_in_y * np.exp(-deltaF_y / thermal_energy) if y_dens > 0 else 0.0         #Droplet to dilute transfer in y
        else:
            k_in_y, k_out_y = 0, 0

        # ---- Phase separation in z ----
        if z_dilt >= z_star or z_dens > 0:
            deltaF_z = free_energy_z(z_dilt + 1, z_bar) - free_energy_z(z_dilt, z_bar)          #change in free energy for z
            k_in_z = (6 * D * z_dilt) / ((Vtot - v * z_dens)**(2./3.))                          #Dilute to droplet transfer in z
            k_out_z = k_in_z * np.exp(-deltaF_z / thermal_energy) if z_dens > 0 else 0.0        #Droplet to dilute tranfer in z
        else:
            k_in_z, k_out_z = 0, 0

        # Repressilator dynamics

        dxdil_prod = alpha_x / (1 + (z_dilt / K_yz)**n)
        dxdilt_dt = dxdil_prod - b_x * x_dilt - k_in_x + k_out_x                                #x_dilt dynamics with phase separation in x
        dxdens_dt = k_in_x - k_out_x - b_x * x_dens                                             #x_dens dynamics with phase separation in x

        dydil_prod = alpha_y / (1 + (x_dilt / K_xy)**n)
        dydilt_dt = dydil_prod - b_y * y_dilt - k_in_y + k_out_y                                #y_dilt dynamics with phase separation in y
        dydens_dt = k_in_y - k_out_y - b_y * y_dens                                             #y_dens dynamics with phase separation in y

        dzdil_prod = alpha_z / (1 + (y_dilt / K_yz)**n)
        dzdilt_dt = dzdil_prod - b_z * z_dilt - k_in_z + k_out_z                                #z_dilt dynamics with phase separation in z
        dzdens_dt = k_in_z - k_out_z - b_z * z_dens                                             #z_dens dynamics with phase separation in z

    return [dxdilt_dt, dxdens_dt, dydilt_dt, dydens_dt, dzdilt_dt, dzdens_dt]

# Initial conditions
y0 = [30, 0.0, 38, 0.0, 31, 0.0]  # x_dilt, x_dens, y_dilt, y_dens, z_dilt, z_dens

t_span = (0, 3000)                                         #Total simulation time

t_eval = np.linspace(t_span[0], t_span[1], 5000)

sol = solve_ivp(repressilator_phase_sep, t_span, y0, t_eval=t_eval, method=  'Radau') #'LSODA')  #'BDF')   'Radau')

if sol.status != 0:
    print(f"Integration failed: {sol.message}")

# Plotting
plt.figure(figsize=(20, 8))
plt.plot(sol.t, sol.y[0], label='x_dilt',color='r')
plt.plot(sol.t, sol.y[2], label='y_dilt',color='b')
plt.plot(sol.t, sol.y[4], label='z_dilt',color='g')
plt.xlabel('Time')
plt.ylabel('Copy Number')
plt.title('Repressilator with Phase Separation in x,  y and z (Mean Field)')
plt.legend()
plt.grid(True)
plt.show()


from google.colab import files

# Extract solution
t = sol.t
x_dilt_t,x_den_t, y_dilt_t, y_dens_t, z_dilt_t, z_dens_t = sol.y

# Save as CSV
np.savetxt("repressilator_phase_x_80_y_80_z_40.csv",
           np.column_stack((t, x_dilt_t, y_dilt_t, z_dilt_t)),
           delimiter=",",
           header="time,x_dilt,y_dilt,z_dilt",
           comments='')

# Download to your system
files.download("repressilator_phase_x_80_y_80_z_40.csv")