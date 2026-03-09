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

Sensitivity analysis for CV_Amplitude and CV_Period
Parameters swept: t_discard, h_min, p_min, d_min, T_min, gamma

Input file format (tab-separated):
  time   X   Y   Z_plus
(or time, x, y, z; the 4th column is treated as Z_+)

Outputs (Origin-friendly .dat):
  sensitivity_<param>.dat
Columns:
  value, X_CV_Amplitude, Y_CV_Amplitude, Zp_CV_Amplitude,
         X_CV_Period,    Y_CV_Period,    Zp_CV_Period,
         X_n_cycles,     Y_n_cycles,     Zp_n_cycles

Plots:
  Inline matplotlib plots for each parameter sweep:
   - CV_Amplitude vs parameter
   - CV_Period vs parameter
   - Detected cycles vs parameter

IMPORTANT:
- Your original amplitude/period extraction function is kept EXACTLY unchanged.
- Sensitivity in gamma requires a wrapper function identical in logic except that
  it replaces the hard-coded 0.8 with gamma. This does not change your original block.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os

# ---------------------------
# User inputs
# ---------------------------
DATA_PATH = "full-time-series-data-Z-PS-100.dat"  # update path if needed
OUT_DIR = "sensitivity_dat_full"

# ---------------------------
# Load time series
# ---------------------------
arr = np.genfromtxt(DATA_PATH, comments="#", delimiter="\t")
if arr.ndim != 2 or arr.shape[1] < 4:
    raise ValueError("Expected at least 4 columns: time, X, Y, Z_+ (tab-separated).")

t = arr[:, 0]
X = arr[:, 1]
Y = arr[:, 2]
Zp = arr[:, 3]  # Z_+ (dilute pool)

# ======================================================
# 1) Extraction function (UNCHANGED; same as your original)
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


# ======================================================
# 2) Wrapper for gamma sweep (identical logic, but uses gamma)
# ======================================================
def amplitude_period_windows_gamma(time, signal, gamma=0.8,
                                   trough_height=-36, trough_prom=15,
                                   trough_dist=400, min_window=60):
    sig = np.array(signal)
    troughs, _ = find_peaks(-sig, height=trough_height,
                            prominence=trough_prom, distance=trough_dist)

    amps, periods = [], []

    for i in range(len(troughs)-1):
        t1, t2 = troughs[i], troughs[i+1]
        if time[t2] - time[t1] < min_window:
            continue

        seg = sig[t1:t2+1]
        peak_idx_rel = np.argmax(seg)
        peak_val = sig[t1+peak_idx_rel]

        thr = gamma * peak_val
        seg_indices = np.arange(t1, t2+1)
        above_thr = seg_indices[sig[t1:t2+1] >= thr]
        if len(above_thr) == 0:
            continue

        L, R = above_thr[0], above_thr[-1]
        mean_high = sig[L:R+1].mean()
        amp = mean_high - sig[t2]

        amps.append(float(amp))
        periods.append(float(time[t2] - time[t1]))

    return amps, periods


# ======================================================
# 3) Defaults and sweeps (edit ranges as needed)
# ======================================================
DEFAULTS = dict(
    t_discard=5000.0,
    h_min=-36,
    p_min=15,
    d_min=400,    # sample-index units
    T_min=60,     # time units
    gamma=0.8
)

SWEEPS = {
    "t_discard": [2000.0, 4000.0, 6000.0, 8000.0, 10000.0, 12000.0],
    "h_min":     [-20, -36, -50, -70],
    "p_min":     [8, 12, 15, 18, 22, 26],
    "d_min":     [200, 300, 400, 500, 650, 800],
    "T_min":     [40, 60, 80, 100, 120],
    "gamma":     [0.7, 0.75, 0.8, 0.85, 0.9],
}

# Proper axis labels (SI-friendly)
XLABELS = {
    "t_discard": r"Transient cutoff, $t_{\mathrm{discard}}$",
    "h_min":     r"Trough depth cutoff (height on $-s(t)$), $h_{\min}$",
    "p_min":     r"Trough prominence threshold, $p_{\min}$",
    "d_min":     r"Minimum trough separation (sample index), $d_{\min}$",
    "T_min":     r"Minimum cycle duration, $T_{\min}$",
    "gamma":     r"Peak-window fraction, $\gamma$",
}


# ======================================================
# 4) Metric computation
# ======================================================
def slice_after_discard(time, signal, t_discard):
    i0 = np.searchsorted(time, t_discard, side="left")
    return time[i0:], signal[i0:]

def safe_cv(vals):
    vals = np.asarray(vals, dtype=float)
    if vals.size < 3:
        return np.nan
    mu = np.mean(vals)
    if mu == 0:
        return np.nan
    return float(np.std(vals, ddof=0) / mu)

def compute_metrics(params):
    t_s, X_s = slice_after_discard(t, X, params["t_discard"])
    _,   Y_s = slice_after_discard(t, Y, params["t_discard"])
    _,   Z_s = slice_after_discard(t, Zp, params["t_discard"])

    out = {}
    for name, sig in [("X", X_s), ("Y", Y_s), ("Z_+", Z_s)]:
        if abs(params["gamma"] - 0.8) < 1e-12:
            amps, periods = amplitude_period_windows(
                t_s, sig, varname=name,
                trough_height=params["h_min"],
                trough_prom=params["p_min"],
                trough_dist=params["d_min"],
                min_window=params["T_min"]
            )
        else:
            amps, periods = amplitude_period_windows_gamma(
                t_s, sig, gamma=params["gamma"],
                trough_height=params["h_min"],
                trough_prom=params["p_min"],
                trough_dist=params["d_min"],
                min_window=params["T_min"]
            )

        out[name] = dict(
            n_cycles=len(amps),
            CV_Amplitude=safe_cv(amps),
            CV_Period=safe_cv(periods)
        )
    return out


# ======================================================
# 5) Export .dat (Origin-friendly)
# ======================================================
def export_sweep_dat(param_name, values, metrics_list):
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"sensitivity_{param_name}.dat")

    header = "\t".join([
        param_name,
        "X_CV_Amplitude", "Y_CV_Amplitude", "Zp_CV_Amplitude",
        "X_CV_Period",    "Y_CV_Period",    "Zp_CV_Period",
        "X_n_cycles",     "Y_n_cycles",     "Zp_n_cycles"
    ])

    mat = []
    for v, m in zip(values, metrics_list):
        mat.append([
            v,
            m["X"]["CV_Amplitude"], m["Y"]["CV_Amplitude"], m["Z_+"]["CV_Amplitude"],
            m["X"]["CV_Period"],    m["Y"]["CV_Period"],    m["Z_+"]["CV_Period"],
            m["X"]["n_cycles"],     m["Y"]["n_cycles"],     m["Z_+"]["n_cycles"],
        ])

    np.savetxt(out_path, np.array(mat, dtype=float), fmt="%.8g", delimiter="\t",
               header=header, comments="")
    print(f"Wrote: {out_path}")


# ======================================================
# 6) Run sweeps, plot inline
# ======================================================
for pname, values in SWEEPS.items():
    values = list(values)
    records = []

    for v in values:
        p = dict(DEFAULTS)
        p[pname] = v
        records.append(compute_metrics(p))

    # Export .dat for Origin
    export_sweep_dat(pname, values, records)

    vals = np.array(values, dtype=float)

    # Build series for plotting
    def series(var, key):
        return np.array([r[var][key] for r in records], dtype=float)

    X_cvA = series("X", "CV_Amplitude")
    Y_cvA = series("Y", "CV_Amplitude")
    Z_cvA = series("Z_+", "CV_Amplitude")

    X_cvT = series("X", "CV_Period")
    Y_cvT = series("Y", "CV_Period")
    Z_cvT = series("Z_+", "CV_Period")

    X_n = series("X", "n_cycles")
    Y_n = series("Y", "n_cycles")
    Z_n = series("Z_+", "n_cycles")

    # Plot CV_Amplitude
    plt.figure()
    plt.plot(vals, X_cvA, marker="o", linewidth=1.5, label="X")
    plt.plot(vals, Y_cvA, marker="^", linewidth=1.5, label="Y")
    plt.plot(vals, Z_cvA, marker="v", linewidth=1.5, label="Z_+")
    plt.xlabel(XLABELS.get(pname, pname))
    plt.ylabel("CV_Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot CV_Period
    plt.figure()
    plt.plot(vals, X_cvT, marker="o", linewidth=1.5, label="X")
    plt.plot(vals, Y_cvT, marker="^", linewidth=1.5, label="Y")
    plt.plot(vals, Z_cvT, marker="v", linewidth=1.5, label="Z_+")
    plt.xlabel(XLABELS.get(pname, pname))
    plt.ylabel("CV_Period")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot detected cycles (diagnostic)
    plt.figure()
    plt.plot(vals, X_n, marker="o", linewidth=1.5, label="X")
    plt.plot(vals, Y_n, marker="^", linewidth=1.5, label="Y")
    plt.plot(vals, Z_n, marker="v", linewidth=1.5, label="Z_+")
    plt.xlabel(XLABELS.get(pname, pname))
    plt.ylabel("Detected cycles")
    plt.legend()
    plt.tight_layout()
    plt.show()