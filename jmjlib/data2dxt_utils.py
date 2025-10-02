"""
data2dxt_utils.py

Utilities for working with Data2D_XT.Data2D objects from jin_pylib.

Author: Joe Mjehovich
Date: 2025-07-12
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

def find_local_peak(DASdata, md, t, half_window=5, use_abs=False, return_index=False):

    """
    Find the local peak amplitude near a specified depth at a given time.

    Parameters
    ----------
    DASdata : Data2D
        A jin_pylib.Data2D_XT.Data2D object with attributes `data`, `taxis`, and `mds`.
    md : float
        Measured depth (center of search window).
    t : float
        Time (in same units as DASdata.taxis).
    half_window : float, optional
        Half-width of the search window in depth units (default: 5).
    use_abs : bool, optional
        If True, find peak based on absolute amplitude.

    Returns
    -------
    peak_amplitude : float
        Amplitude at the peak location.
    peak_depth : float
        Depth corresponding to the peak.
    
    Raises
    ------
    ValueError
        If `md` is outside the range of the DASdata.mds.
    """

    # get index of nearest time
    time_idx = np.argmin(np.abs(DASdata.taxis - t))

    # search window depths
    lower = md - half_window
    upper = md + half_window

    mask = (DASdata.mds >= lower) & (DASdata.mds <= upper)

    if not np.any(mask):
        raise ValueError('ERROR: md is outside range.')

    search_depths = DASdata.mds[mask]
    amplitudes = DASdata.data[mask, time_idx]

    if use_abs:
        amplitudes = np.abs(amplitudes)

    peak_idx = np.argmax(amplitudes)
    peak_amplitude = amplitudes[peak_idx]
    peak_depth = search_depths[peak_idx]

    if return_index:
        full_index = np.where(mask)[0][peak_idx]
        return peak_amplitude, peak_depth, full_index
    else:
        return peak_amplitude, peak_depth

def compute_spectrogram(DASdata, md, fs=None, nperseg=256, noverlap=None):
    dt = np.median(np.diff(DASdata.taxis))
    fs = fs or 1 / dt

    # Find closest depth index
    md_idx = np.argmin(np.abs(DASdata.mds - md))
    chan_data = DASdata.data[md_idx, :]

    # Compute spectrogram
    f, t, Sxx = spectrogram(chan_data, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return f, t, Sxx

def plot_spectrogram(DASdata, md, fs=None, nperseg=256, noverlap=None, log_scale=True,
                     clim_percentiles=(5,99)):
    f, t, Sxx = compute_spectrogram(DASdata, md, fs=fs, nperseg=nperseg, noverlap=noverlap)
    Sxx_plot = 10 * np.log10(Sxx + 1e-12) if log_scale else Sxx

    vmin = np.percentile(Sxx_plot, clim_percentiles[0])
    vmax = np.percentile(Sxx_plot, clim_percentiles[1])

    plt.figure(figsize=(8,5))
    plt.pcolormesh(t, f, Sxx_plot, shading='gouraud')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title(f"Spectrogram at Depth ≈ {md}")
    plt.clim(vmin, vmax)
    plt.colorbar(label='Power (dB)' if log_scale else 'Amplitude')
    plt.tight_layout()
    plt.show()