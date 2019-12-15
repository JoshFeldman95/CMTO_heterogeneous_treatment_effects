import numpy as np
from sklearn.linear_model import LinearRegression


class HeterogeneousTreatmentEffectModel:
    def _init__(self):
        raise NotImplementedError

    def fit(X, t, y):
        raise NotImplementedError

    def fit_IV(X, t, z, y):
        raise NotImplementedError

    def predict(X):
        raise NotImplementedError

    def predict_IV(X, z):
        raise NotImplementedError

class LinearRegressionHTE(HeterogeneousTreatmentEffectModel):
    def __init__(self, *args, **kwargs):
        self.model = LinearRegression(*args, **kwargs)

    def fit(self, X, t, y):
        X_with_treatment_interactions = np.concatenate((X, X * t), axis = 1)
        self.model.fit(X_with_treatment_interactions, y)

    def predict(self, X):
        # t = 0
        X_with_treatment_0_interactions = np.concatenate((X, X * 0), axis = 1)
        y_0 = self.model.predict(X_with_treatment_0_interactions)

        # t = 1
        X_with_treatment_1_interactions = np.concatenate((X, X * 1), axis = 1)
        y_1 = self.model.predict(X_with_treatment_1_interactions)
        return y_1 - y_0
