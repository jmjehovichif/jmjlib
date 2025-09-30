"""
polynomial_fit.py

A lightweight tool for fitting and evaluating polynomial models (including linear)
using NumPy. Encapsulates polyfit, prediction, scoring, and plotting.

Author: Joe Mjehovich
Date: 2025-07-12
"""

import numpy as np
import matplotlib.pyplot as plt

class PolynomialFitting():

    """
    Fit and evaluate polynomial models using NumPy.

    This class wraps NumPy's polyfit/polyval methods to provide a reusable interface
    for curve fitting, prediction, and diagnostics including residual plots and R².

    Parameters
    ----------
    xdata : array_like, optional
        Independent variable values.
    ydata : array_like, optional
        Dependent variable values.
    degree : int, optional
        Degree of the polynomial to fit. Default is 1 (linear).

    Attributes
    ----------
    coeffs : ndarray or None
        Polynomial coefficients after fitting, or None if not fit yet.
    degree : int
        Degree of the polynomial model.
    xdata : ndarray
        Stored x values.
    ydata : ndarray
        Stored y values.
    """

    def __init__(self, xdata=None, ydata=None, degree=1):
        self.xdata = np.array(xdata) if xdata is not None else None
        self.ydata = np.array(ydata) if ydata is not None else None
        self.coeffs = None
        self.degree = degree

    def fit(self):
        self.coeffs = np.polyfit(x=self.xdata, y=self.ydata, deg=self.degree)

    def predict(self, x=1):
        if self.coeffs is None:
            raise ValueError("Model must be fit before making predictions.")
        return np.polyval(self.coeffs, np.array(x))
    
    def plot_fit(self):
        x = np.linspace(self.xdata.min(), self.xdata.max(), 100)
        y_pred = self.predict(x)

        plt.figure(figsize=(8,6))
        plt.plot(self.xdata, self.ydata, 'o', c='r', label='Data')
        plt.plot(x, y_pred, label='Fitted Line')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()

    def set_data(self, xdata, ydata):
        self.xdata = np.array(xdata)
        self.ydata = np.array(ydata)
        self.coeffs = None

    def score(self):
        if self.coeffs is None:
            raise ValueError("No model has been fit.")
        y_pred = self.predict(self.xdata)
        sst = np.sum((self.ydata - np.mean(self.ydata))**2)
        sse = np.sum((self.ydata - y_pred)**2)
        return 1 - (sse/sst)
    
    def plot_residuals(self):
        if self.coeffs is None:
            raise ValueError("No model has been fit.")
        y_pred = self.predict(self.xdata)
        residuals = self.ydata - y_pred
        plt.figure(figsize=(8,6))
        plt.scatter(self.xdata, residuals, color='b', label='Residuals')
        plt.axhline(y=0, xmin=0, xmax=1, linestyle='--', color='r')
        plt.xlabel('x')
        plt.ylabel('Residuals (y - y_pred)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()
        
    def get_params(self):
        return self.coeffs