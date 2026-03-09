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



Toggle Switch with Phase Separation in Y (Stochastic Simulation)
=================================================================

This script simulates a toggle switch where only transcription factor
Y undergoes *phase separation* into dilute and dense phases:

    X;   Y_dilt, Y_dens

The system exhibits bistability — one TF dominates while repressing the other.
Phase separation modulates switching noise and stability.

A parameter sweep is performed over different φ* values (phase separation thresholds)
for Y, and the following statistics are computed for each case:

    • Mean
    • Variance
    • Fano factor (Var / Mean)
    • Coefficient of variation (CV = σ / Mean)

Outputs:
    - x_low.csv,  x_high.csv
    - y_low.csv,  y_high.csv
Each file contains statistics for the low and high states across φ* combinations.
"""

import numpy as np

# Random seed
np.random.seed(0)


# ==============================
# Parameter sweep setup
# ==============================
phi_star_values = [0.00085, 0.0008, 0.00075, 0.0007, 0.00065, 0.0006, 0.00055]
thresh = 38.2
kB = 1.38e-23
T = 305
thermal_energy = kB * T

# ==============================
# Function to run one simulation
# ==============================
def run_simulation(phi_star_y, thresh=38.2):
    # Fixed toggle switch parameters
    alpha_x, alpha_y = 5, 5                     #synthesis rate
    tau_x, tau_y = 1/0.05, 1/0.05               #lifetime
    b_x, b_y = 1/tau_x, 1/tau_y                 # Degradation rate
    n = 2                                       #cooperativity
    kxy, kyx = 30, 30                           #Dissociation constant

    v = 1.0e-25                                 #volume of a TF
    volume_factor = 1e5   
    Vtot = volume_factor * v                    #Total volume
    tau_d = 0.1 * tau_y                         #Diffusion time
    D = Vtot**(2. / 3.) / (6. * tau_d)          #Diffusion coefficient

    y_star = phi_star_y * volume_factor         #threshold for y phase separation
    mu_y = -thermal_energy * (np.log(phi_star_y) - phi_star_y)           #Chemical potential

    # --------------------------
    # Free energy functions
    # --------------------------
    def free_energy_y(y_dilt, y_bar):
        if y_dilt == 0:
            return 0.0
        denom = volume_factor - y_bar + y_dilt
        temp1 = y_dilt / denom
        temp1 = np.clip(temp1, 1e-12, None)
        term1 = -mu_y * (y_bar - y_dilt)
        term2 = thermal_energy * y_dilt * (np.log(temp1) - 1)
        return term1 + term2

    # --------------------------
    # Simulation loop
    # --------------------------
    t = 0.0
    tend = 100000 * tau_y           #total simulation time
    
    #initial values
    x = 0
    y_dilt, y_dens = 0, 0
    
    #store
    t_list = []
    x_list, y_dilt_list = [], []

    while t <= tend:
        y_bar = y_dilt + y_dens

        if y_dilt < y_star and y_dens == 0:          #no phase separation case
            a1 = alpha_x / (1 + (y_dilt / kyx) ** n)
            a2 = b_x * x
            a3 = alpha_y / (1 + (x / kxy) ** n)
            a4 = b_y * y_dilt
            a0 = a1 + a2 + a3 + a4
            if a0 == 0: break
            r1 = np.random.rand()
            tau = (1.0 / a0) * np.log(1.0 / r1)
            r2 = np.random.rand() * a0
            if r2 < a1:
                x += 1                                            #x synthesis
            elif r2 < a1 + a2:
                x -= 1                                            #x degradation
            elif r2 < a1 + a2 + a3:
                y_dilt += 1                                       #y synthesis
            else:
                y_dilt -= 1                                       #y degradation
        else:                                                     #when Y phase separates
            dF_y = free_energy_y(y_dilt + 1, y_bar) - free_energy_y(y_dilt, y_bar)
            k_in_y = (6. * D * y_dilt) / ((Vtot - v * y_dens)**(2. / 3.)) if y_dilt > y_star else 0           # diffusion limited transport from y dilute to droplet phase
            k_out_y = k_in_y * np.exp(-dF_y / thermal_energy) if y_dens > 0 else 0

            a1 = alpha_x / (1 + (y_dilt / kyx) ** n)
            a2 = b_x * x
            a3 = alpha_y / (1 + (x / kxy) ** n)
            a4 = b_y * y_dilt
            a5 = k_in_y
            a6 = k_out_y
            a7 = b_y * y_dens

            a0 = a1 + a2 + a3 + a4 + a5 + a6 + a7
            if a0 == 0: break

            r1 = np.random.rand()
            tau = (1.0 / a0) * np.log(1.0 / r1)
            r2 = np.random.rand() * a0

            if r2 < a1: x += 1                                         #x synthesis
            elif r2 < a1 + a2: x -= 1                                  #x degradation
            elif r2 < a1 + a2 + a3: y_dilt += 1                        #y dilute synthesis
            elif r2 < a1 + a2 + a3 + a4: y_dilt -= 1                   #y dilute degradation
            elif r2 < a1 + a2 + a3 + a4 + a5: y_dilt -= 1; y_dens += 1 #y dilute --> y droplet
            elif r2 < a1 + a2 + a3 + a4 + a5 + a6: y_dilt += 1; y_dens -= 1 #y droplet --> y dilute
            else: y_dens -= 1                                          #y dense degradation

        t += tau
        t_list.append(t)
        x_list.append(x)
        y_dilt_list.append(y_dilt)

    return np.array(t_list), np.array(x_list), np.array(y_dilt_list)


def compute_stats(data):
    """Compute mean, variance, CV, Fano"""
    if len(data) == 0:
        return [np.nan, np.nan, np.nan, np.nan]
    mean_val = np.mean(data)
    var_val = np.var(data)
    cv_val = np.std(data) / mean_val if mean_val != 0 else np.nan
    fano_val = var_val / mean_val if mean_val != 0 else np.nan
    return [mean_val, var_val, fano_val, cv_val]

# Prepare result lists for each mode
x_low_results = []
x_high_results = []
y_low_results = []
y_high_results = []


for phi_star_y in phi_star_values:
    # Run simulation
    t_arr, x_arr, y_dilt_arr = run_simulation(phi_star_y, thresh)

    # Remove transient (keep only data after t >= 2000)
    mask = t_arr >= 2000
    x_arr = x_arr[mask]
    y_dilt_arr = y_dilt_arr[mask]

    # Split data by threshold
    x_low  = x_arr[x_arr < thresh]
    x_high = x_arr[x_arr >= thresh]
    y_low  = y_dilt_arr[y_dilt_arr < thresh]
    y_high = y_dilt_arr[y_dilt_arr >= thresh]

    # Convert φ* → threshold copynumber
    y_threshold = round(phi_star_y * 1e5, 1)

    # Collect results (x_threshold is not applicable for 1PS case, set to 0 or NaN)
    x_threshold = 0  # Since X doesn't phase separate in 1PS

    x_low_results.append([x_threshold, y_threshold] + compute_stats(x_low))
    x_high_results.append([x_threshold, y_threshold] + compute_stats(x_high))
    y_low_results.append([x_threshold, y_threshold] + compute_stats(y_low))
    y_high_results.append([x_threshold, y_threshold] + compute_stats(y_high))

# Save each as separate CSV
header = "x_threshold,y_threshold,mean,var,fano,cv"

np.savetxt("x_low.csv",  np.array(x_low_results, dtype=object), fmt="%s", delimiter=",", header=header, comments="")
np.savetxt("x_high.csv", np.array(x_high_results, dtype=object), fmt="%s", delimiter=",", header=header, comments="")
np.savetxt("y_low.csv",  np.array(y_low_results, dtype=object), fmt="%s", delimiter=",", header=header, comments="")
np.savetxt("y_high.csv", np.array(y_high_results, dtype=object), fmt="%s", delimiter=",", header=header, comments="")