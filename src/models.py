import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures

from keras.models import Sequential
from keras.layers import Dense, Activation

import bartpy.sklearnmodel

# define base S-Learner and T-Learners
class SLearner:
    def __init__(self, model, *args, **kwargs):
        self.model = model

    def fit(self, X, t, y):
        X_with_treatment = np.concatenate((X, t.reshape(-1, 1)), axis = 1)
        self.model.fit(X_with_treatment, y)

    def predict(self, X):
        # t = 0
        X0 = np.concatenate((X, np.zeros((X.shape[0], 1))) , axis = 1)
        y0 = self.model.predict(X0)

        # t = 1
        X1 = np.concatenate((X, np.ones((X.shape[0], 1))), axis = 1)
        y1 = self.model.predict(X1)
        return y1 - y0

class TLearner:
    def __init__(self, model0, model1):
        self.model0 = model0
        self.model1 = model1

    def fit(self, X, t, y):
        X1 = X[t == 1]
        y1 = y[t == 1]
        self.model1.fit(X1, y1)

        X0 = X[t == 0]
        y0 = y[t == 0]
        self.model0.fit(X0, y0)

    def predict(self, X):
        return self.model1.predict(X) - self.model0.predict(X)

# implement models
class LinearRegression_TLearner(TLearner):
    def __init__(self, *args, **kwargs):
        self.model0 = LinearRegression(*args, **kwargs)
        self.model1 = LinearRegression(*args, **kwargs)

class LassoWithInteractions_TLearner(TLearner):
    def __init__(self, *args, **kwargs):
        self.model0 = LassoWithInteractions(*args, **kwargs)
        self.model1 = LassoWithInteractions(*args, **kwargs)

class RandomForest_SLearner(SLearner):
    def __init__(self, *args, **kwargs):
        self.model = RandomForestRegressor(*args, **kwargs)

class RandomForest_TLearner(TLearner):
    def __init__(self, *args, **kwargs):
        self.model0 = RandomForestRegressor(*args, **kwargs)
        self.model1 = RandomForestRegressor(*args, **kwargs)

class DecisionTree_SLearner(SLearner):
    def __init__(self, *args, **kwargs):
        self.model = DecisionTreeRegressor(*args, **kwargs)

class DecisionTree_TLearner(TLearner):
    def __init__(self, *args, **kwargs):
        self.model0 = DecisionTreeRegressor(*args, **kwargs)
        self.model1 = DecisionTreeRegressor(*args, **kwargs)

# Implement Wrappers
class LassoWithInteractions():
    def __init__(self, degree = 2, *args, **kwargs):
        self.poly = PolynomialFeatures(degree=degree)
        self.model = Lasso(*args, **kwargs)

    def fit(self, X, y):
        X = self.poly.fit_transform(X)
        self.model.fit(X, y)

    def predict(self, X):
        X = self.poly.fit_transform(X)
        return self.model.predict(X)
