"""
multivariate_polyfit.py

A reusable Python class for fitting multivariate polynomial models using 
scikit-learn. Supports multiple input features (X) and single or multi-output
targets (Y). Designed for applications such as GPS/fiber path modeling, 
well trajectory fitting, and other multivariate regression tasks.

Author: Joe Mjehovich
Date: 2025-10-03
"""

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

class MultivariatePolynomialFitting:
    def __init__(self, X=None, Y=None, degree=1):
        """
        X: array-like, shape (n_samples, n_features)
        Y: array-like, shape (n_samples,) or (n_samples, n_outputs)
        degree: polynomial degree (int)
        """
        self.X = np.array(X) if X is not None else None
        self.Y = np.array(Y) if Y is not None else None
        self.degree = int(degree)
        self.poly = None
        self.model = None

    def set_data(self, X, Y):
        """Store data and clear any previous model."""
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.poly = None
        self.model = None

    def fit(self):
        """Fit PolynomialFeatures -> LinearRegression"""
        if self.X is None or self.Y is None:
            raise ValueError("Call set_data(X, Y) before fit().")
        
        self.poly = PolynomialFeatures(degree=self.degree)
        X_poly = self.poly.fit_transform(self.X)

        self.model = LinearRegression()
        self.model.fit(X_poly, self.Y)

    def predict(self, X_new):
        """Predict Y for new X. Return shape (n_samples,) or (n_samples, n_outputs)."""
        if self.model is None or self.poly is None:
            raise ValueError("Model not fit yet.")
        X_new = np.array(X_new)
        X_new_poly = self.poly.transform(X_new)
        return self.model.predict(X_new_poly)

    def score(self):
        """Return R^2 score on the training data (multioutput handled by sklearn)."""
        if self.model is None:
            raise ValueError("Model not fit yet.")
        X_poly = self.poly.transform(self.X)
        return self.model.score(X_poly, self.Y)