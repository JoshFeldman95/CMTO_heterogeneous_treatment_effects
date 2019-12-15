import unittest

from src import models
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# load data
df = pd.read_csv('./data/cmto.csv', index_col = 0)
df = df.dropna()

# prep data
T = df['received_cmto_services'].values.reshape(-1, 1)
Z = df['treatment_group'].values.reshape(-1, 1)
Y = (df['forecast_kravg30_p25'].values * 100).reshape(-1, 1)

X_col = ['pha', 'hoh_age', 'child_count', 'child_age', 'speaks_english',
     'born_abroad', 'working', 'homeless', 'hh_income', 'origin_pop2010',
     'black', 'white', 'asian','latino', 'less_hs', 'college_plus', 'origin_forecast_kravg30_p25']
X_df = df[X_col]

columns_to_scale = ['hoh_age', 'child_age', 'hh_income', 'origin_pop2010']
scaler = StandardScaler()
X_df.loc[:,columns_to_scale] = scaler.fit_transform(X_df[columns_to_scale])

X = X_df.values

# create train and test data
X_train, X_test, T_train, T_test, Y_train, Y_test, Z_train, Z_test = train_test_split(X, T, Y, Z, test_size=0.2)


def test_models():
    model_list = [models.LinearRegressionHTE()]
    for model in model_list:
        print(X_train.shape, T_train.shape, Y_train.shape)
        model.fit(X_train, T_train, Y_train)
        assert model.predict(X_test).shape == Y_test.shape

if __name__ == '__main__':
    unittest.main()
