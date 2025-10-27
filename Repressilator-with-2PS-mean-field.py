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



Repressilator with Phase Separation of Two Transcription Factors (Mean-Field)
============================================================================

This script simulates the canonical *repressilator* motif in which three
transcription factors (TFs) repress each other cyclically:

    X ⊣ Y ⊣ Z ⊣ X

Here, *phase separation* is explicitly introduced for **two TFs (Y and Z)**.
Each can exist in two subpopulations:
    - Dilute (cytoplasmic) phase: y_dilt, z_dilt
    - Dense (condensed) phase:   y_dens, z_dens

Feedback repression acts through the *dilute* components only:
    z_dilt ⊣ x,  x ⊣ y_dilt,  y_dilt ⊣ z_dilt

The model integrates mean-field ODEs to track deterministic dynamics and
captures both transcriptional feedback and diffusion-controlled phase
exchange between dilute and condensed compartments.

Outputs:
    • Time evolution of x, y_dilt, z_dilt
    • CSV file: "repressilator_phsp_y_80_z_40.csv"
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from google.colab import files

# Parameters
alpha_x, alpha_y, alpha_z = 5, 5, 5                                             #synthesis rate
tau_x, tau_y, tau_z = 1/0.02, 1/0.02, 1/0.02                                    #TF lifetime
b_x, b_y, b_z = 1/tau_x, 1/tau_y, 1/tau_z                                       #Degradation rate constants
n = 3                                                                           #Cooperativity
K_xy, K_yz, K_zx = 20, 20, 20                                                   #Dissociation constant

# Phase separation constants
v = 1.0e-25                                                                     #volume of a TF molecule
volume_factor = 1e5
Vtot = volume_factor * v                                                        #total volume of the system
phi_star_y = 0.0008
phi_star_z = 0.0004
y_star = phi_star_y*volume_factor                                               #threshold for y phase separation
z_star = phi_star_z*volume_factor                                               #threshold for z phase separation
thermal_energy = 1.38e-23 * 305                                                 #Thermal energy = K_B *T
mu_y = -thermal_energy * (np.log(phi_star_y) - phi_star_y)                      #chemical potential for y
mu_z = -thermal_energy * (np.log(phi_star_z) - phi_star_z)                      #chemical potential for z
tau_d = 0.1 *tau_z                                                              #Diffusion time
D = Vtot**(2. / 3.) / (6. * tau_d)                                              #Diffusion coefficient

# Free energy function
def free_energy_y(dilt, bar):                                                   #define free energy function for y
    eps = 1e-12
    denom = max(volume_factor - (bar - dilt), eps)
    temp1 = dilt / denom
    temp1 = max(temp1, eps)
    term1 = -mu_y * (bar - dilt)
    term2 = thermal_energy * dilt * (np.log(temp1) - 1)
    return term1 + term2

# Free energy function
def free_energy_z(dilt, bar):                                                   #define free energy function for z
    eps = 1e-12
    denom = max(volume_factor - (bar - dilt), eps)
    temp1 = dilt / denom
    temp1 = max(temp1, eps)
    term1 = -mu_z * (bar - dilt)
    term2 = thermal_energy * dilt * (np.log(temp1) - 1)
    return term1 + term2

# Repressilator with phase separation in y and z
def repressilator_phase_sep(t, y):
    x, y_dilt, y_dens, z_dilt, z_dens = y
    x, y_dilt, y_dens, z_dilt, z_dens = map(lambda v: max(v, 0), (x, y_dilt, y_dens, z_dilt, z_dens))

    # Total TF levels
    y_bar = y_dilt + y_dens
    z_bar = z_dilt + z_dens

    # Condition 1: Before either phase separates
    if y_dilt < y_star and y_dens == 0 and z_dilt < z_star and z_dens == 0:
        dxdt = alpha_x / (1 + (z_dilt / K_zx)**n) - b_x * x                                   #Here TF x do not phase separates. x gets feedback from dilute phase of z, i.e., z_dilt
        dydilt_dt = alpha_y / (1 + (x / K_xy)**n) - b_y * y_dilt                              #Before the y copy number exceeds the phase separation threshold, no terms ralated to dilute/droplet exchange
        dzdilt_dt = alpha_z / (1 + (y_dilt / K_yz)**n) - b_z * z_dilt                         #Before the z copy number exceeds the phase separation threshold, no terms related to dilute/dropltet exchange
        dydens_dt = 0                                                                         #no y droplet formation before phase separation of y
        dzdens_dt = 0                                                                         #no z droplet formation before phase separation of z

    else:
        # ---- Phase separation in y ----
        if y_dilt >= y_star or y_dens > 0:
            deltaF_y = free_energy_y(y_dilt + 1, y_bar) - free_energy_y(y_dilt, y_bar)        #change in free energy function
            k_in_y = (6 * D * y_dilt) / ((Vtot - v * y_dens)**(2./3.))                        #Dilute to droplet transfer of y
            k_out_y = k_in_y * np.exp(-deltaF_y / thermal_energy) if y_dens > 0 else 0.0      #Droplet to dilute transfer of y
        else:
            k_in_y, k_out_y = 0, 0

        # ---- Phase separation in z ----
        if z_dilt >= z_star or z_dens > 0:
            deltaF_z = free_energy_z(z_dilt + 1, z_bar) - free_energy_z(z_dilt, z_bar)       #change in free energy function
            k_in_z = (6 * D * z_dilt) / ((Vtot - v * z_dens)**(2./3.))                       #Dilute to droplet transfer in z
            k_out_z = k_in_z * np.exp(-deltaF_z / thermal_energy) if z_dens > 0 else 0.0     #Droplet to dilute transfer in z
        else:
            k_in_z, k_out_z = 0, 0

        # Repressilator dynamics
        dxdt = alpha_x / (1 + (z_dilt / K_zx)**n) - b_x * x                                #x dynamics same as before as here x do not phase separates

        dydil_prod = alpha_y / (1 + (x / K_xy)**n)
        dydilt_dt = dydil_prod - b_y * y_dilt - k_in_y + k_out_y                           # y_dilt dynamics
        dydens_dt = k_in_y - k_out_y - b_y * y_dens                                        # y_dens dynamics

        dzdil_prod = alpha_z / (1 + (y_dilt / K_yz)**n)
        dzdilt_dt = dzdil_prod - b_z * z_dilt - k_in_z + k_out_z                           # z_dilt dynamics
        dzdens_dt = k_in_z - k_out_z - b_z * z_dens                                        # z_dens dynamics

    return [dxdt, dydilt_dt, dydens_dt, dzdilt_dt, dzdens_dt]

# Initial conditions
y0 = [30.0, 38.0, 0.0, 31.0, 0.0]              #(x, y_dilt,y_dens,z_dilt,z_dens)

#simulation time
t_span = (0, 3000)
t_eval = np.linspace(t_span[0], t_span[1], 5000)



#solve
sol = solve_ivp(repressilator_phase_sep, t_span, y0, t_eval=t_eval, method=  'Radau') #'LSODA')  #'BDF')   'Radau')

if sol.status != 0:
    print(f"Integration failed: {sol.message}")

# Plotting
plt.figure(figsize=(20, 6))
plt.plot(sol.t, sol.y[0], label='x', color='r')
plt.plot(sol.t, sol.y[1], label='y_dilt',color= 'b')
plt.plot(sol.t, sol.y[3], label='z_dilt',color='g')
plt.xlabel('Time')
plt.ylabel('Copy Number')
plt.xlim(0,3000)
plt.ylim(0,140)
plt.title('Repressilator with Phase Separation in y and z (Mean Field)')
plt.legend()
plt.show()


# Extract solution
t = sol.t
x_t, y_dilt_t, y_dens_t, z_dilt_t, z_dens_t = sol.y

# Save as CSV
np.savetxt("repressilator_phsp_y_80_z_40.csv",
           np.column_stack((t, x_t, y_dilt_t, z_dilt_t)),
           delimiter=",",
           header="time,x,y_dilt,z_dilt",
           comments='')

# Download to your system
files.download("repressilator_phsp_y_80_z_40.csv")