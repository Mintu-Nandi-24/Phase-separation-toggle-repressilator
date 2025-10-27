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



Repressilator with Phase Separation of a Single Transcription Factor (Mean-Field)
================================================================================

This script simulates a canonical repressilator—a cyclic gene network
where three transcription factors (TFs) repress one another:

    X ⊣ Y ⊣ Z ⊣ X

Here only TF Z undergoes **phase separation**, splitting into:
    - Dilute phase: z_dilt
    - Dense (condensed) phase: z_dens

Regulatory feedback occurs through the dilute component (z_dilt),
which represses x, while x represses y, and y represses z_dilt.

The system is described by mean-field ordinary differential equations (ODEs)
that include both transcriptional regulation and diffusion-controlled
exchange between the dilute and condensed phases.

Outputs:
    • Time evolution of x, y, and z_dilt
    • CSV file: "repressilator_phsp_z_40.csv"
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import mpl_toolkits.mplot3d
from google.colab import files

# Parameters
alpha_x, alpha_y, alpha_z = 5, 5, 5            # Synthesis rates
tau_x, tau_y, tau_z = 1/0.02, 1/0.02, 1/0.02   # lifetime
b_x, b_y, b_z = 1/tau_x, 1/tau_y, 1/tau_z      # Degradation rates
n = 3                                          # coopearativity
K_xy, K_yz, K_zx = 20, 20, 20                  # Dissociation constant

# Phase separation constants
v = 1.0e-25                                                   #volume of one TF
volume_factor = 1e5
Vtot = volume_factor * v                                      #Total volume of the cell
phi_star =   0.0004
x_star = phi_star * volume_factor                             #phase separation threshold for x
thermal_energy = 1.38e-23 * 305                               #thermal energy= K_B*T
mu = -thermal_energy * (np.log(phi_star) - phi_star)          #Chemical potential
tau_d = 0.1 *tau_z                                            #Diffusion time
D = Vtot**(2. / 3.) / (6. * tau_d)                            #Diffusion coefficient


def free_energy(z_dilt, z_bar):                                                 #Define free energy function
    eps = 1e-12  # small epsilon to avoid zero division
    denominator = max(volume_factor - (z_bar - z_dilt), eps)  # clamp#

    temp1 = z_dilt / denominator
    temp1 = max(temp1, eps)  # avoid log(0)

    term1 = -mu * (z_bar - z_dilt)
    term2 = thermal_energy * z_dilt * (np.log(temp1) - 1)

    return term1 + term2



def repressilator_with_phase_sep(t, y):
    x, y_, z_dilt, z_dens = y  # unpack variables


    x = max(x, 0)
    y_ = max(y_, 0)
    z_dilt = max(z_dilt, 0)
    z_dens = max(z_dens, 0)



    # Condition 1: Before phase separation
    if z_dilt < x_star and z_dens == 0:
        dxdt = alpha_x / (1 + (z_dilt / K_zx)**n) - b_x * x
        dydt = alpha_y / (1 + (x / K_xy)**n) - b_y * y_
        dzdilt_dt = alpha_z / (1 + (y_ / K_yz)**n) - b_z * z_dilt
        dzdens_dt = 0                                                                                       # no dense phase yet

    # Condition 2: After phase separation starts
    else:
        # Total z
        z_bar = z_dilt + z_dens



       # Free energy change for phase separation
        delta_F = free_energy(z_dilt + 1, z_bar) - free_energy(z_dilt, z_bar)                         #Change in free energy

        # Phase separation rates
        k_in = (6. * D * z_dilt) / ((Vtot - v * z_dens)**(2. / 3.))                                   #Dilute to droplet tranfer in z
        k_out = k_in * np.exp(-delta_F / thermal_energy) if z_dens > 0 else 0.0                       #Droplet to dilute transfer in z
        # Core repression dynamics
        dxdt = alpha_x / (1 + (z_dilt / K_zx)**n) - b_x * x
        dydt = alpha_y / (1 + (x / K_xy)**n) - b_y * y_

        # z production and degradation
        dzdil_prod = alpha_z / (1 + (y_ / K_yz)**n)
        dzdil_deg = b_z * z_dilt
        zdens_deg = b_z * z_dens



        # Final derivatives
        dzdilt_dt = dzdil_prod - dzdil_deg - k_in + k_out
        dzdens_dt = k_in - k_out - zdens_deg

    dydt = [dxdt, dydt, dzdilt_dt, dzdens_dt]

    # Ensure non-negativity for all components
  #  return [max(0, val) for val in dydt]
    return dydt


# Initial conditions and time span
y0 = [30.0, 38.0, 31.0, 0.0]  # (x, y, z_dilt, z_dens)
t_span = (0, 2500)
t_eval = np.linspace(t_span[0], t_span[1], 5000)

# Solve ODE using stiff solver
sol = solve_ivp(repressilator_with_phase_sep, t_span, y0, t_eval=t_eval, method='Radau')#, BDF, rtol=1e-8, atol=1e-12)

if sol.status != 0:
    print(f"Integration failed: {sol.message}")


# Plot results
plt.figure(figsize=(20,6))
plt.plot(sol.t, sol.y[0], label='x',color='r')
plt.plot(sol.t, sol.y[1], label='y',color='b')
plt.plot(sol.t, sol.y[2], label='z_dilt',color='g')
plt.xlabel('Time')
plt.ylabel('Copy number')
plt.xlim(0,2500)
plt.ylim(0,140)
plt.title('Mean-field Repressilator with Phase Separation in z')
plt.legend()
plt.show()



# Extract solution
t = sol.t
x_t, y_t, z_dilt_t, z_dens_t = sol.y

# Save as CSV
np.savetxt("repressilator_phsp_z_40.csv",
           np.column_stack((t, x_t, y_t, z_dilt_t)),
           delimiter=",",
           header="time,x,y,z_dilt",
           comments='')

# Download to your system
files.download("repressilator_phsp_z_40.csv")