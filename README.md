===============================================
Title: Phase separation as a tunable regulator of canonical gene regulatory motifs
===============================================

Authors:
    Priya Chakraborty$
        The Institute of Mathematical Sciences, CIT Campus, Taramani,
        Chennai 600113, India

    Mintu Nandi*,$
        Universal Biology Institute, The University of Tokyo,
        7-3-1 Hongo, Bunkyo-ku, Tokyo 113-0033, Japan

    Sandeep Choubey*
        The Institute of Mathematical Sciences, CIT Campus, Taramani,
        Chennai 600113, India
        Homi Bhabha National Institute, Training School Complex,
        Anushaktinagar, Mumbai 400094, India

* Corresponding authors (mintunandi@ubi.s.u-tokyo.ac.jp, sandeep@imsc.res.in)
$ Equal contributions

-------------------------------------------------------------------------------
Overview
-------------------------------------------------------------------------------
This repository contains all simulation and mean-field analysis codes used in the study  
**"Phase separation as a tunable regulator of canonical gene regulatory motifs."**

The scripts implement both deterministic (mean-field ODE) and stochastic (Gillespie) simulations  
of canonical gene regulatory motifs — including the **toggle switch** and **repressilator** —  
extended to account for **phase separation** of transcription factors (TFs).

Each model variant (canonical, one-phase, two-phase, and three-phase) corresponds to a specific  
combination of TFs undergoing condensate formation. The goal is to understand how compartmentalization  
modulates steady-state stability, oscillatory dynamics, and variability.

-------------------------------------------------------------------------------
Repository Organization
-------------------------------------------------------------------------------

### 1. Canonical Toggle Switch
| Script | Description |
|--------|--------------|
| `Canonical-toggle-mean-field.py` | Deterministic mean-field ODE analysis of the toggle switch motif (X ⊣ Y, Y ⊣ X). Computes nullclines, fixed points, eigenvalues, and separatrix to identify bistability and phase-plane topology. |
| `Canonical-toggle-simulation.py` | Gillespie simulation of the toggle switch to quantify stochastic switching between the two stable states. Exports trajectory data and stationary distributions. |

---

### 2. Canonical Repressilator
| Script | Description |
|--------|--------------|
| `Canonical-repressilator-mean-field.py` | Deterministic ODE-based model of the canonical repressilator (X ⊣ Y ⊣ Z ⊣ X). Computes oscillatory trajectories and steady-state properties. |
| `Canonical-repressilator-simulation.py` | Gillespie simulation of the canonical repressilator to capture stochastic oscillations and quantify amplitude and period fluctuations. |

---

### 3. Toggle Switch with Phase Separation
| Script | Description |
|--------|--------------|
| `Toggle-with-1PS-mean-field.py` | Mean-field ODE model where only one transcription factor (X) undergoes phase separation, forming dilute and dense subpopulations. Captures how compartmentalization alters bistability. |
| `Toggle-with-1PS-simulation.py` | Gillespie simulation of the toggle switch with phase separation of one TF. Quantifies switching noise, dwell times, and occupancy probabilities. |
| `Toggle-with-2PS-mean-field.py` | Mean-field ODE model where oboth TFs (X and Y)) undergoes phase separation, forming dilute and dense subpopulations. Captures how compartmentalization alters bistability. |
| `Toggle-with-2PS-simulation.py` | Stochastic simulation where both TFs (X and Y) can phase separate. Performs a parameter sweep over phase-separation thresholds (φ*_X, φ*_Y) to compute mean, variance, Fano factor, and CV for both stable states. |

---

### 4. Repressilator with Phase Separation
| Script | Description |
|--------|--------------|
| `Repressilator-with-1PS-mean-field.py` | Mean-field ODE model where only Z undergoes phase separation. Tracks time evolution of (x, y, z_dilt) and exports data as `repressilator_phsp_z_40.csv`. |
| `Repressilator-with-1PS-simulation.py` | Gillespie simulation of the repressilator with one phase-separating TF (Z). Calculates amplitude and period statistics across varying thresholds n*_Z. |
| `Repressilator-with-2PS-mean-field.py` | Mean-field ODE model with phase separation of Y and Z. Incorporates diffusion-mediated exchange between dilute and dense phases. Exports `repressilator_phsp_y_80_z_40.csv`. |
| `Repressilator-with-2PS-simulation.py` | Gillespie simulation where Y and Z undergo phase separation. Quantifies mean, variance, and CV of oscillation amplitude and period. |
| `Repressilator-with-3PS-mean-field.py` | Mean-field model where all three TFs (X, Y, Z) phase separate. Tracks coupled ODEs for dilute and dense components and exports time evolution data. |
| `Repressilator-with-3PS-simulation.py` | Gillespie simulation for three-phase repressilator (X, Y, Z). Analyzes joint effects of (n*_X, n*_Y, n*_Z) on oscillation coherence and variability. |

---

-------------------------------------------------------------------------------
Dependencies
-------------------------------------------------------------------------------
All scripts use standard Python scientific libraries:
- `numpy`
- `matplotlib`
- `scipy` (for `solve_ivp`, `fsolve`, `find_peaks`)
- `mpl_toolkits.mplot3d` (for 3D phase visualization, optional)
- `google.colab.files` (optional, for CSV downloads in Colab)

Ensure Python ≥3.8 with these packages installed via:
```bash
pip install numpy matplotlib scipy
