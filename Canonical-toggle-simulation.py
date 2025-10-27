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



Canonical Toggle Switch (Gillespie Simulation)
==============================================

This script simulates the canonical *toggle switch* using the
stochastic Gillespie algorithm. Two transcription factors (TFs),
X and Y, mutually repress each other:

    X ⊣ Y  and  Y ⊣ X

The model captures intrinsic noise due to molecular fluctuations.
No phase separation or additional sequestration mechanisms are present.

Outputs:
    • Time trajectories of X and Y
    • CSV file: "toggle_switch_trajectory.csv"
"""

# Import libraries

import numpy as np
import matplotlib.pyplot as plt
import os

# Parameters

alpha_x, alpha_y = 5, 5           # Synthesis rate
n = 2                             # Cooperativity of binding
K_xy, K_yx = 30, 30               # Dissociation constant


# Lifetime of TFs
tau_x = 1.0 / 0.05        # x lifetime
tau_y = 1.0 / 0.05       # y lifetime

#Degradation
b_x = 1 / tau_x           # degradation rate of x
b_y = 1 / tau_y           # degradation rate of y



#store
x_count = []
y_count = []
time_points = []


#simulation time
tend = 20000* tau_y

#Initial copy number
t = 0.0
x = 0.0
y = 0.0

#Start gilliespie simulation

while t <= tend:

        a1 = alpha_x / (1 + (y/K_yx)**n)
        a2 = b_x * x
        a3 = alpha_y / (1 + (x/K_xy)**n)
        a4 = b_y * y                                 #y degradation
        a0 = a1 + a2 + a3 + a4                       #sum of propensity

        if a0 == 0:
            break

        r1 = np.random.uniform(0, 1)
        tau = (1.0 / a0) * np.log(1.0 / r1)
        r2 = np.random.uniform(0, 1) * a0
        if r2 < a1:
            x += 1                                  #x synthesis
        elif r2 < a1 + a2:
            x -= 1                                  #x degradation
        elif r2 < a1 + a2 + a3:
            y += 1                                  #y synthesis
        else:
            y -= 1                                  #y degradation
        t += tau


        x_count.append(x )
        y_count.append(y )
        time_points.append(t)



# Convert to numpy array
data = np.column_stack([time_points, x_count, y_count])


#Remove transient part

filtered_data = data[data[:, 0] > 2500]

time_points = filtered_data[:, 0]
x_count = filtered_data[:, 1]
y_count = filtered_data[:, 2]






#plot the trajectory
plt.figure(figsize=(10, 6))
plt.plot(time_points, x_count,label='x',color='r')
plt.plot(time_points, y_count, label = 'y',color ='b')
plt.xlabel('Time')
plt.ylabel('Copy Number')
plt.title(f'Toggle switch time trajectory')
plt.legend()
plt.show()


# --- Save data and download in Colab ---

from google.colab import files

# Define output file name
file_name = "toggle_switch_trajectory.csv"

# Combine the data columns (time, x, y)
data = np.column_stack((time_points, x_count, y_count))

# Save as CSV-style .dat file
np.savetxt(
    file_name,
    data,
    delimiter=",",
    header="time,x,y",
    fmt="%.5f",
    comments=''
)

# Download the file in system
files.download(file_name)

print(f"Data successfully saved to: {file_name}")