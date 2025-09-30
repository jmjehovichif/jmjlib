"""
trace.py

This module defines the Trace class, a simple wrapper around 1D signals
with support for time alignment, normalization, peak detection, FFT plotting,
and basic signal analysis.

Author: Joe Mjehovich
Date: 2025-07-11
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, hilbert

class Trace:
    
    """
    A class for representing and analyzing 1D time-aligned traces.

    Parameters
    ----------
    data : array_like
        The trace values.
    time : array_like, optional
        The corresponding time vector. If not provided, assumes uniform
        sampling with time = [0, 1, ..., len(data)-1].

    Raises
    ------
    ValueError
        If `data` and `time` have different lengths.
    """

    def __init__(self, data, time=None):
        self.data = np.array(data)
        self.time = time if time is not None else np.arange(len(self.data))

        if len(self.data) != len(self.time):
            raise ValueError("Data and time arrays must be the same length.")
        
    def get_duration(self):
        return self.time[-1] - self.time[0]
    
    def get_rms(self):
        return np.sqrt(np.sum(self.data**2) / len(self.data))
    
    def normalize(self):
        self.data = self.data / np.max(np.abs(self.data))

    def standardize(self, return_new=False):
        mean = np.mean(self.data)
        std = np.std(self.data) + 1e-12

        standardized = (self.data - mean) / std

        if return_new:
            return Trace(standardized, self.time)
        else:
            self.data = standardized

    def get_window(self, start, end=None):
        if start < 0 or start >= len(self.data):
            raise ValueError("Start index out of bounds")
        if end is not None and (end < 0 or end > len(self.data)):
            raise ValueError("End index out of bounds")

        if end is None:
            return self.data[:start]
        else:
            return self.data[start:end]
        
    def get_window_by_time(self, start, end=None):
        if end is None:
            mask = self.time >= start
        else:
            mask = (self.time >= start) & (self.time <= end)

        return self.time[mask], self.data[mask]
        
    def plot_window_by_time(self, start, end=None):
        t_win, x_win = self.get_window_by_time(start, end)
        plt.figure(figsize=(8,6))
        plt.plot(t_win, x_win)
        plt.title(f"Windowed Trace: {start} to {end if end else "end"}")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    def compute_fft(self, max_freq=None, logscale=False):
        dt = np.median(np.diff(self.time))
        fs = 1/dt
        sp = np.fft.fft(self.data)
        freq = np.fft.fftfreq(self.time.shape[-1], d=dt)

        mag = np.abs(sp)

        if max_freq is not None:
            mask = (freq > 0) & (freq <= max_freq)
        else:
            mask = freq > 0
        mag = mag[mask]
        freq = freq[mask]

        if logscale:
            mag = 20 * np.log10(mag + 1e-12)

        return freq, mag

    def plot_fft(self, max_freq=None, logscale=False):
        freq, mag = self.compute_fft(max_freq=max_freq, logscale=logscale)
        plt.figure(figsize=(8,6))
        plt.plot(freq, mag)
        plt.title("FFT Spectrum")
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude (dB)' if logscale else "Amplitude")
        plt.grid(True, linestyle = '--', alpha=0.5)
        plt.show()

    def detect_peaks(self, height=None, distance=None, prominence=None):
        self.peaks, self.props = find_peaks(self.data, height=height, distance=distance, prominence=prominence)

        return self.peaks

    def plot_peaks(self):
        if not hasattr(self, "peaks") or self.peaks is None:
            raise ValueError("No peaks. Run detect_peaks() first.")
        
        plt.figure(figsize=(8,6))
        plt.plot(self.time, self.data, label='Data')
        plt.plot(self.time[self.peaks], self.data[self.peaks],"x", color='tab:orange', label='Peaks')
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.title("Detected Peaks")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    def derivative(self):
        dy = np.gradient(self.data, self.time)
        return Trace(dy, time=self.time)
    
    def instantaneous_frequency(self, smooth_phase: bool = True):
        """
        Returns (envelope, inst_freq) as Trace objects or arrays.

        Parameters
        ----------
        smooth_phase : bool
            Whether to unwrap the phase before differentiating to avoid jumps.

        Returns
        -------
        envelope : Trace
        inst_freq : Trace
            Instantaneous frequency in Hz.
        """
        analytic_signal = hilbert(self.data)
        envelope = np.abs(analytic_signal)
        if smooth_phase:
            inst_phase = np.unwrap(np.angle(analytic_signal))
        else:
            inst_phase = np.angle(analytic_signal)

        if np.any(np.diff(self.time) <= 0):
            raise ValueError("Time vector must be strictly increasing for instantaneous frequency.")

        inst_freq = np.gradient(inst_phase, self.time) / (2 * np.pi)

        return Trace(envelope, time=self.time), Trace(inst_freq, time=self.time)