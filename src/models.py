import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures

from keras.models import Model
from keras.layers import Input, Dense, Activation
from keras.optimizers import Adam
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

class MLP_SLearner(SLearner):
    def __init__(self, *args, **kwargs):
        self.model = MLP(*args, **kwargs)

class MLP_TLearner(TLearner):
    def __init__(self, *args, **kwargs):
        self.model0 = MLP(*args, **kwargs)
        self.model1 = MLP(*args, **kwargs)

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

class MLP():
    def __init__(self, num_features = 17, num_layers = 4, hidden_dim = 32,
                 epochs = 20, batch_size = 32, lr = 1e-4):
        input = Input(shape=(num_features,))
        x = Dense(hidden_dim, activation = 'relu')(input)

        for _ in range(num_layers - 1):
            x = Dense(hidden_dim, activation = 'relu')(x)

        out = Dense(1)(x)
        self.model = Model(inputs=input, outputs=out)
        opt = Adam(learning_rate=lr)
        self.model.compile(opt, loss = 'mse')

        self.batch_size = batch_size
        self.epochs = epochs

    def fit(self, X, y):
        self.model.fit(X, y, batch_size = self.batch_size, epochs = self.epochs)

    def predict(self, X):
        return self.model.predict(X).reshape(-1)
