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



Stochastic Simulation of Canonical Repressilator using the Gillespie Algorithm
==============================================================================

This script performs an exact stochastic simulation (Gillespie algorithm)
of the canonical *repressilator*, a cyclic gene regulatory motif where three
transcription factors (TFs) repress one another:

    X ⊣ Y ⊣ Z ⊣ X

Each TF is synthesized and degraded with rates that depend on its repressor
through Hill-type regulation. The stochastic approach captures intrinsic
noise arising from finite copy numbers and random molecular events.

Outputs:
    • Time evolution of X, Y, and Z molecule counts
    • CSV file: "Canonical_repressilator_Gillespie_trajectory.csv"
"""




import numpy as np
import matplotlib.pyplot as plt

# Define constants
alpha_x, alpha_y, alpha_z = 5, 5, 5               #Synthesis rate constant
n= 3                                              #Cooperativity
K_xy, K_yz, K_zx = 20, 20, 20                     #Dissociation constant
tau_x, tau_y, tau_z = 1/0.02, 1/0.02, 1/0.02      #lifetime
b_x, b_y, b_z = 1/tau_x, 1/tau_y, 1/tau_z         #Degradation constant


# Initial number of molecules
x = 30
y = 38
z = 31

# Simulation time
t_final = 10000.0

# Lists to store results
time_points = [0]
x_counts = [x]
y_counts = [y]
z_counts = [z]


# Gillespie algorithm
t = 0.0
while t < t_final:

    # Calculate propensities
    a1 = alpha_x/(1+(z/K_zx)**n)
    a2 = b_x*x
    a3 = alpha_y/(1+(x/K_xy)**n)
    a4 = b_y*y
    a5 = alpha_z/(1+(y/K_yz)**n)
    a6 = b_z*z

    a0 = a1+a2+a3+a4+a5+a6            # Total propensity



# Handle the case when a0 is zero to avoid ZeroDivisionError
    if a0 == 0:
        break                                 # Exit the loop if no reactions can occur

    # Time to next reaction
    r1 = np.random.uniform(0, 1)
    tau = (1.0 / a0) * np.log(1.0 / r1)

    # Determine which reaction occurs
    r2 = np.random.uniform(0, 1) * a0
    if r2 < a1:
        x += 1                                #x synthesis
    elif r2 < a1 + a2:
        x -= 1                                # x dedradation
    elif r2 < a1 + a2 + a3:
        y += 1                                # y synthesis
    elif r2 < a1 + a2 + a3 + a4:
        y -= 1                                # y degradation
    elif r2 < a1 + a2 + a3 + a4 + a5:
        z += 1                                # z synthesis
    else:
        z -= 1                                # z degradation

    # Update time
    t += tau

    # Record results
    time_points.append(t)
    x_counts.append(x)
    y_counts.append(y)
    z_counts.append(z)


data= np.column_stack([time_points, x_counts, y_counts, z_counts])

#keep only data after transient (t>5000)
filtered_data = data[data[:, 0] > 5000]

time_points = filtered_data[:, 0]
x_counts = filtered_data[:, 1]
y_counts = filtered_data[:, 2]
z_counts = filtered_data[:, 3]



# Plot results
plt.figure(figsize=(20, 5))
#plt.plot(time_points, mRNA_counts, label='mRNA')
plt.plot(time_points, x_counts, label='x', color='r')
plt.plot(time_points, y_counts, label='y',color='b')
plt.plot(time_points, z_counts, label='z',color='g')
plt.xlabel('Time')
plt.ylabel('TF Copy number')
plt.legend()
plt.show()


#np.savetxt(
#    "Cannonical_repressilator_Gilliespie_trajectory.dat",
#    filtered_data,
#    header="time\tTF_x\tTF_y\tTF_z",
#    fmt="%.5f",
#    delimiter="\t"
#)
#print("Trajectory saved to Cannonical_repressilator_Gilliespie_trajectory.dat")



from google.colab import files
np.savetxt(
    "Cannonical_repressilator_Gilliespie_trajectory.csv",
    filtered_data,
    delimiter=",",
    header="time,x,y,z",
    fmt="%.5f",
    comments=''
)
files.download("Cannonical_repressilator_Gilliespie_trajectory.csv")