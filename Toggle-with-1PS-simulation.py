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



Toggle Switch with Phase Separation in Y (Stochastic Gillespie Simulation)
==========================================================================

This script simulates a **toggle switch** where two transcription factors (TFs)
mutually repress each other:

    X ⊣ Y and Y ⊣ X

Only TF **Y** undergoes *phase separation*, existing in:
    - Dilute phase:  y_dilt
    - Dense (condensed/droplet) phase:  y_dens

Regulation occurs through **y_dilt** (the soluble/dilute fraction),
while y_dens evolves via diffusion-limited molecular exchange
controlled by a free-energy difference between the phases.

Phase separation introduces stochastic sequestration of Y molecules,
which affects the switching dynamics between the two stable states.

Outputs:
    • Stochastic trajectories of X and Y (dilute)
    • CSV file: “toggle_trajectory_phsp_y_65.csv”
"""

import numpy as np

# Parameters

alpha_x = 5                                               # Synthesis of  x
alpha_y =5                                                #synthesis of y
n = 2                                                     # Cooperativity of binding
K_xy = 30                                                 #Dissociation constant
K_yx = 30                                                 #Dissociation constant

# Lifetime of TFs
tau_x = 1.0 / 0.05                                        # x lifetime
tau_y = 1.0 / 0.05                                        # y lifetime

b_x = 1/tau_x                                            # x degradation rate
b_y = 1/tau_y                                            # y degradation rate

# System parameters
v = 1.0 * 10**-25                                        # Volume of one molecule
volume_factor = 10**5
Vtot = volume_factor * v                                # Total volume
kB = 1.38 * 10**-23                                     # Boltzmann constant
T = 305                                                 # Temperature
thermal_energy = kB * T
phi_star = 0.00065                                       # Saturation volume fraction, change the value of phi_star here
y_star = phi_star * volume_factor                       # threshold for y phase separation
mu = -thermal_energy * (np.log(phi_star) - phi_star)    # chemical potential
tau_d = 0.1 * tau_y                                     # Diffusion time
D = Vtot**(2. / 3.) / (6. * tau_d)                      # Diffusion coefficient

# Free energy function
def free_energy(y_dilt, y_bar):
    term1 = -mu*(y_bar-y_dilt)
    temp1 = y_dilt / (volume_factor - y_bar + y_dilt)
    temp1 = np.clip(temp1, 1e-12, None)
    term2 = thermal_energy * y_dilt * (np.log(temp1) - 1)
    return term1 + term2


#store values
x_count = []
y_dilute_count = []
y_dense_count = []
total_protein_count = []
time_points = []

#Initial values
k_in = 0.0
k_out = 0.0
t = 0.0
x = 0
y_dilt = 0
y_dens = 0

#Total simulation time
tend = 100000* tau_y                                         #100000*tau_y             #increase the simulation time here,

while t <= tend:
    y_bar = y_dilt + y_dens
    if y_dilt < y_star and y_dens == 0:                                  # when y dilute is less than y_star there is no phase separation in y, ydens==0
        a1 = alpha_x / (1 + (y_dilt/K_yx)**n)                            #x synthesis
        a2 = b_x * x                                                     #x dedgadation
        a3 = alpha_y / (1 + (x/K_xy)**n)                                 #y dilute synthesis
        a4 = b_y * y_dilt                                                #y dilute degradation
        a0 = a1 + a2 + a3 + a4

        if a0 == 0:
            break

        r1 = np.random.uniform(0, 1)
        tau = (1.0 / a0) * np.log(1.0 / r1)
        r2 = np.random.uniform(0, 1) * a0
        if r2 < a1:
            x += 1                                                      #x synthesis
        elif r2 < a1 + a2:
            x -= 1                                                      #x degradation
        elif r2 < a1 + a2 + a3:
            y_dilt += 1                                                 #y dilute synthesis
        else:
            y_dilt -= 1                                                 #y dilute degradation
        t += tau
    else:                                                                                       # when y_dilt is greater than y_star, y phase separates
        free_energy_change = free_energy(y_dilt + 1, y_bar) - free_energy(y_dilt, y_bar)
        k_in = (6. * D * y_dilt) / ((Vtot - v * y_dens)**(2. / 3.))                             # y dilute to y droplet transfer
        k_out = k_in * np.exp(-free_energy_change / thermal_energy) if y_dens != 0 else 0.0     # y droplet to y dilute transfer

        a1 = alpha_x / (1 + (y_dilt/K_yx)**n)                                                   # x synthesis
        a2 = b_x * x                                                                            # x degradation
        a3 = alpha_y / (1 + (x/K_xy)**n)                                                        # y dilute synthesis
        a4 = b_y * y_dilt                                                                        # y dilute degradation
        a5 = k_in                                                                               #y dilute to y droplet transfer
        a6 = k_out                                                                              #y droplet to y dilute transfer
        a7 = b_y * y_dens                                                                       #y droplet degradation
        a0 = a1 + a2 + a3 + a4 + a5 + a6 + a7                                                   #Total propensity

        if a0 == 0:
            break

        r1 = np.random.uniform(0, 1)
        tau = (1.0 / a0) * np.log(1.0 / r1)
        r2 = np.random.uniform(0, 1) * a0
        if r2 < a1:
            x += 1                                                                           #x synthesis
        elif r2 < a1 + a2:
            x -= 1                                                                           #x degradation
        elif r2 < a1 + a2 + a3:
            y_dilt += 1                                                                      #y dilute synthesis
        elif r2 < a1 + a2 + a3 + a4:
            y_dilt -= 1                                                                      #y dilute degradation
        elif r2 < a1 + a2 + a3 + a4 + a5:
            y_dilt -= 1                                                                      #y dilute-->y droplet
            y_dens += 1
        elif r2 < a1 + a2 + a3 + a4 + a5 + a6:
            y_dilt += 1                                                                      #y droplet-->y dilute
            y_dens -= 1
        else:
            y_dens -= 1                                                                      #y droplet degradation
        t += tau

    x_count.append(x )
    y_dilute_count.append(y_dilt )
    y_dense_count.append(y_dens )
    time_points.append(t)


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

# Plot x over time
plt.plot(time_points, x_count, label='x', color='red')

# Plot y dilute over time
plt.plot(time_points, y_dilute_count, label=' y (dilute)', color='blue')

plt.xlabel('Time')
plt.ylabel('Copy Number')
plt.title('Time Series of x and y (Dilute)')
plt.legend()
plt.grid(True)
plt.show()

#import csv                                                                     #to save trajectory data when not working in colab

# Define the output filename
#output_filename = "toggle_trajectory_phsp_y_65.csv"

# Open the file and write data
#with open(output_filename, mode='w', newline='') as file:
#    writer = csv.writer(file)

 #   # Write header
 #   writer.writerow([
 #       "time_points",
 #       "x", "y_dilute"
 #   ])

    # Write rows
 #   for i in range(len(time_points)):
 #       writer.writerow([
 #           time_points[i],
 #           x_count[i],
 #           y_dilute_protein_count[i],
 #
 #       ])

#print(f"\nSimulation data saved to '{output_filename}'")


import numpy as np
from google.colab import files

# Stack data into a 2D array
data = np.column_stack([
    time_points,
    x_count,
    y_dilute_count

])

# Keep only rows where time_points > 2000
filtered_data = data[data[:, 0] > 2000]

# Save filtered data as CSV
output_filename = "toggle_trajectory_phsp_y_65.csv"
np.savetxt(
    output_filename,
    filtered_data,
    delimiter=",",
    header="time_points,x,y_dilute",
    comments='',
    fmt="%.5f"
)

# Download the file
files.download(output_filename)