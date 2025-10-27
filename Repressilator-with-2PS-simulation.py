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

-------------------------------------------------------------------------------
Description:
This script implements Gillespie simulations of a canonical repressilator
in which two transcription factors (Y and Z) undergo phase separation.
The objective is to quantify how their phase-separation thresholds (n*_Y, n*_Z)
jointly modulate oscillatory behavior—specifically the amplitude and period
variability of X, Y, and Z in the dilute phase.

For each pair of threshold values (n*_Y, n*_Z), the simulation generates
stochastic trajectories of all three transcription factors, identifies
oscillation windows between consecutive troughs, and computes the mean,
variance, and coefficient of variation (CV) for both amplitude and period.

-------------------------------------------------------------------------------
Outputs:
    - amp-data-x.dat, amp-data-y.dat, amp-data-z.dat
      → Mean and variance of oscillation amplitudes for X, Y, and Z

    - amp-cv-x.dat, amp-cv-y.dat, amp-cv-z.dat
      → Coefficients of variation (CV) of amplitudes

    - period-data-x.dat, period-data-y.dat, period-data-z.dat
      → Mean and variance of oscillation periods

    - period-cv-x.dat, period-cv-y.dat, period-cv-z.dat
      → CV of oscillation periods
-------------------------------------------------------------------------------
Notes:
    • Amplitude and period are computed via trough-based window analysis
      using local minima (find_peaks on −signal).
    • All kinetic parameters are chosen within physiological ranges
      consistent with Elowitz & Leibler (Nature 2000) and Bennett et al. (PNAS 2007).
    • The model can be extended to one-phase (only Z) or three-phase (X, Y, Z)
      cases for comparative analysis.


"""

import numpy as np
# import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ======================================================
# 1. Define amplitude/period detection function
# ======================================================
def amplitude_period_windows(time, signal, varname="X",
                             trough_height=-36, trough_prom=15,
                             trough_dist=400, min_window=60):
    sig = np.array(signal)
    troughs, _ = find_peaks(-sig, height=trough_height,
                            prominence=trough_prom, distance=trough_dist)

    amps, periods, times = [], [], []

    for i in range(len(troughs)-1):
        t1, t2 = troughs[i], troughs[i+1]
        if time[t2] - time[t1] < min_window:
            continue

        seg = sig[t1:t2+1]
        peak_idx_rel = np.argmax(seg)
        peak_val = sig[t1+peak_idx_rel]

        thr = 0.8 * peak_val
        seg_indices = np.arange(t1, t2+1)
        above_thr = seg_indices[sig[t1:t2+1] >= thr]
        if len(above_thr) == 0:
            continue

        L, R = above_thr[0], above_thr[-1]
        mean_high = sig[L:R+1].mean()
        amp = mean_high - sig[t2]

        amps.append(float(amp))
        periods.append(float(time[t2] - time[t1]))
        times.append(time[t2])   # record trough time

    return amps, periods


# -------------------------------
# Gillespie simulation function
# -------------------------------
def run_simulation(n_star_y, n_star_z):
    
    # -----------------------------
    # Fixed system parameters
    # -----------------------------
    a_x, a_y, a_z = 5, 5, 5
    b_x, b_y, b_z = 1, 1, 1
    c_x, c_y, c_z = 0.02, 0.02, 0.02
    n = 3
    kxy, kyz, kzx = 20, 20, 20

    # Phase separation constants
    v = 1.0e-25
    volume_factor = 1e5
    Vtot = volume_factor * v
    thermal_energy = 1.38e-23 * 305  # k_B * T
    
    phi_star_y = n_star_y/volume_factor
    phi_star_z = n_star_z/volume_factor
    
    # Chemical potentials
    mu_y = -thermal_energy * (np.log(phi_star_y) - phi_star_y)
    mu_z = -thermal_energy * (np.log(phi_star_z) - phi_star_z)
    
    # Timescales
    tau_y, tau_z = 1 / c_y, 1 / c_z
    taud_y, taud_z = 0.1 * tau_y, 0.1 * tau_z
    D_y = Vtot ** (2. / 3.) / (6. * taud_y)
    D_z = Vtot ** (2. / 3.) / (6. * taud_z)
    
    # Initial conditions
    simulation_time = 0.0
    x = 30
    y_dilt, y_dens = 38, 0.0
    z_dilt, z_dens = 31, 0.0
    
    tend=1000000
    
    # -----------------------------
    # Functions
    # -----------------------------
    def free_energy(phi_star, mu, dilt, bar):
        term1 = -mu * (bar - dilt)
        temp1 = dilt / (volume_factor - bar + dilt)
        temp1 = np.clip(temp1, 1e-12, None)
        term2 = thermal_energy * dilt * (np.log(temp1) - 1)
        return term1 + term2
    
    
    recorded_times, x_count, y_dil_count, z_dil_count = [], [], [], []

    while simulation_time < tend:
        
        # Record
        x_count.append(x)
        y_dil_count.append(y_dilt)
        z_dil_count.append(z_dilt)
        recorded_times.append(simulation_time)
        
        # Totals
        y_bar, z_bar = y_dilt + y_dens, z_dilt + z_dens

        # Free energy changes
        dG_y = free_energy(phi_star_y, mu_y, y_dilt + 1, y_bar) - free_energy(phi_star_y, mu_y, y_dilt, y_bar)
        dG_z = free_energy(phi_star_z, mu_z, z_dilt + 1, z_bar) - free_energy(phi_star_z, mu_z, z_dilt, z_bar)

        # Phase separation kinetics
        k_in_y = (6. * D_y * y_dilt) / ((Vtot - v * y_dens) ** (2. / 3.)) if y_dilt > n_star_y else 0
        k_out_y = k_in_y * np.exp(-dG_y / thermal_energy) if y_dens > 0 else 0

        k_in_z = (6. * D_z * z_dilt) / ((Vtot - v * z_dens) ** (2. / 3.)) if z_dilt > n_star_z else 0
        k_out_z = k_in_z * np.exp(-dG_z / thermal_energy) if z_dens > 0 else 0

        # Reactions
        a1 = a_x / (b_x + (z_dilt / kzx) ** n); a2 = c_x * x

        a3 = a_y / (b_y + (x / kxy) ** n); a4 = c_y * y_dilt
        a5 = k_in_y; a6 = k_out_y; a7 = c_y * y_dens

        a8 = a_z / (b_z + (y_dilt / kyz) ** n); a9 = c_z * z_dilt
        a10 = k_in_z; a11 = k_out_z; a12 = c_z * z_dens

        a0 = sum([a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12])
        if a0 == 0: break
        
        # Gillespie step
        r1 = np.random.uniform()
        tau = (1.0 / a0) * np.log(1.0 / r1)
        simulation_time += tau
        r2 = np.random.uniform() * a0
        cumulative = np.cumsum([a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12])

        if r2 < cumulative[0]:
            x += 1
        elif r2 < cumulative[1]:
            x -= 1
        elif r2 < cumulative[2]:
            y_dilt += 1
        elif r2 < cumulative[3]:
            y_dilt -= 1
        elif r2 < cumulative[4]:
            y_dilt -= 1; y_dens += 1
        elif r2 < cumulative[5]:
            y_dilt += 1; y_dens -= 1
        elif r2 < cumulative[6]:
            y_dens -= 1
        elif r2 < cumulative[7]:
            z_dilt += 1
        elif r2 < cumulative[8]:
            z_dilt -= 1
        elif r2 < cumulative[9]:
            z_dilt -= 1; z_dens += 1
        elif r2 < cumulative[10]:
            z_dilt += 1; z_dens -= 1
        else:
            z_dens -= 1
        
    
    # Remove transient (e.g., t < 5000)
    transient_end_idx = next((i for i, t in enumerate(recorded_times) if t >= 5000), 0)
    x_count = x_count[transient_end_idx:]
    y_count = y_dil_count[transient_end_idx:]
    z_count = z_dil_count[transient_end_idx:]
    recorded_times_sliced = recorded_times[transient_end_idx:]
    
    return (np.array(recorded_times_sliced),
            np.array(x_count),
            np.array(y_count),
            np.array(z_count))



amp_data_x = []
amp_cv_x = []
amp_data_y = []
amp_cv_y = []
amp_data_z = []
amp_cv_z = []

period_data_x = []
period_cv_x = []
period_data_y = []
period_cv_y = []
period_data_z = []
period_cv_z = []

n_list = [80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130]

index = 0

for n_star_y_in in n_list:
    n_star_y = n_star_y_in
    
    for n_star_z_in in n_list:
        n_star_z = n_star_z_in
        
        t, x, y, z = run_simulation(n_star_y, n_star_z)
        
        amps_x, periods_x = amplitude_period_windows(t, x, varname="X")
        amps_y, periods_y = amplitude_period_windows(t, y, varname="Y")
        amps_z, periods_z = amplitude_period_windows(t, z, varname="Z")
        
        
        mean_amp_x = np.mean(amps_x)
        var_amp_x = np.var(amps_x)
        cv_amp_x = np.sqrt(var_amp_x)/mean_amp_x
        
        mean_amp_y = np.mean(amps_y)
        var_amp_y = np.var(amps_y)
        cv_amp_y = np.sqrt(var_amp_y)/mean_amp_y
        
        mean_amp_z = np.mean(amps_z)
        var_amp_z = np.var(amps_z)
        cv_amp_z = np.sqrt(var_amp_z)/mean_amp_z
        
        mean_period_x = np.mean(periods_x)
        var_period_x = np.var(periods_x)
        cv_period_x = np.sqrt(var_period_x)/mean_period_x
        
        mean_period_y = np.mean(periods_y)
        var_period_y = np.var(periods_y)
        cv_period_y = np.sqrt(var_period_y)/mean_period_y
        
        mean_period_z = np.mean(periods_z)
        var_period_z = np.var(periods_z)
        cv_period_z = np.sqrt(var_period_z)/mean_period_z
        
        amp_data_x.append((n_star_y, n_star_z, mean_amp_x, var_amp_x))
        amp_cv_x.append((n_star_y, n_star_z, cv_amp_x))
        
        amp_data_y.append((n_star_y, n_star_z, mean_amp_y, var_amp_y))
        amp_cv_y.append((n_star_y, n_star_z, cv_amp_y))
        
        amp_data_z.append((n_star_y, n_star_z, mean_amp_z, var_amp_z))
        amp_cv_z.append((n_star_y, n_star_z, cv_amp_z))
        
        period_data_x.append((n_star_y, n_star_z, mean_period_x, var_period_x))
        period_cv_x.append((n_star_y, n_star_z, cv_period_x))
        
        period_data_y.append((n_star_y, n_star_z, mean_period_y, var_period_y))
        period_cv_y.append((n_star_y, n_star_z, cv_period_y))
        
        period_data_z.append((n_star_y, n_star_z, mean_period_z, var_period_z))
        period_cv_z.append((n_star_y, n_star_z, cv_period_z))
        
        index += 1
        print(index, n_star_y, n_star_z) 



# --- Export data ---
np.savetxt("amp-data-x.dat", np.array(amp_data_x), header="y\tz\tmean\tvar", fmt="%d\t%d\t%.6f\t%.6f", delimiter="\t")
np.savetxt("amp-cv-x.dat", np.array(amp_cv_x), header="y\tz\tcv", fmt="%d\t%d\t%.6f", delimiter="\t")

np.savetxt("amp-data-y.dat", np.array(amp_data_y), header="y\tz\tmean\tvar", fmt="%d\t%d\t%.6f\t%.6f", delimiter="\t")
np.savetxt("amp-cv-y.dat", np.array(amp_cv_y), header="y\tz\tcv", fmt="%d\t%d\t%.6f", delimiter="\t")

np.savetxt("amp-data-z.dat", np.array(amp_data_z), header="y\tz\tmean\tvar", fmt="%d\t%d\t%.6f\t%.6f", delimiter="\t")
np.savetxt("amp-cv-z.dat", np.array(amp_cv_z), header="y\tz\tcv", fmt="%d\t%d\t%.6f", delimiter="\t")

np.savetxt("period-data-x.dat", np.array(period_data_x), header="y\tz\tmean\tvar", fmt="%d\t%d\t%.6f\t%.6f", delimiter="\t")
np.savetxt("period-cv-x.dat", np.array(period_cv_x), header="y\tz\tcv", fmt="%d\t%d\t%.6f", delimiter="\t")

np.savetxt("period-data-y.dat", np.array(period_data_y), header="y\tz\tmean\tvar", fmt="%d\t%d\t%.6f\t%.6f", delimiter="\t")
np.savetxt("period-cv-y.dat", np.array(period_cv_y), header="y\tz\tcv", fmt="%d\t%d\t%.6f", delimiter="\t")

np.savetxt("period-data-z.dat", np.array(period_data_z), header="y\tz\tmean\tvar", fmt="%d\t%d\t%.6f\t%.6f", delimiter="\t")
np.savetxt("period-cv-z.dat", np.array(period_cv_z), header="y\tz\tcv", fmt="%d\t%d\t%.6f", delimiter="\t")


