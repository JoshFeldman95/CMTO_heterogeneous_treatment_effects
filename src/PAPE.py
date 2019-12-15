import models
import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_col = ['pha', 'hoh_age', 'child_count', 'child_age', 'speaks_english',
     'born_abroad', 'working', 'homeless', 'hh_income', 'origin_pop2010',
     'black', 'white', 'asian','latino', 'less_hs', 'college_plus', 'origin_forecast_kravg30_p25']

def PAPE(Y, T, ITR):
    assert Y.shape[0] == T.shape[0]
    assert Y.shape[0] == ITR.shape[0]

    n = Y.shape[0]
    n1 = np.sum(T)
    n0 = n - n1

    p = np.sum(ITR)/n

    pop_avg_value_ITR = (
        1 / n1 * np.sum(Y * T * ITR)
        + 1 / n0 * np.sum(Y * (1 - T) * (1 - ITR))
    )

    pop_avg_value_random = (
        p / n1 * np.sum(Y * T)
        + (1 - p) / n0 * np.sum(Y * (1 - T))
    )

    PAPE = n / (n - 1) * (pop_avg_value_ITR - pop_avg_value_random)
    return PAPE


def get_data():
    # load data
    df = pd.read_csv('./data/cmto.csv', index_col = 0)
    df = df.dropna()

    # prep data
    T = df['received_cmto_services'].values
    Z = df['treatment_group'].values
    Y = df['forecast_kravg30_p25'].values * 100

    X_df = df[X_col]

    columns_to_scale = ['hoh_age', 'child_age', 'hh_income', 'origin_pop2010']
    scaler = StandardScaler()
    X_df.loc[:,columns_to_scale] = scaler.fit_transform(X_df[columns_to_scale])

    X = X_df.values
    return X, T, Y, Z

def get_budget_ITR(HTE_pred, num_treated):
    # budge fit_transform
    HTE_pred = HTE_pred + np.random.normal(scale = 1e-10, size = HTE_pred.shape)
    sorted_idx = np.argsort(HTE_pred.reshape(-1))
    threshold = HTE_pred[sorted_idx[-(num_treated)]]
    ITR_budget = HTE_pred >= threshold
    print(sum(ITR_budget))
    print(num_treated)
    assert sum(ITR_budget) == num_treated
    return ITR_budget

if __name__ == '__main__':
    X, T, Y, Z = get_data()

    # create train and test data
    X_train, X_test, T_train, T_test, Y_train, Y_test, Z_train, Z_test = (
        train_test_split(X, T, Y, Z, test_size=0.2)
    )

    model_list = [
        models.LinearRegression_TLearner(),
        models.LassoWithInteractions_TLearner(),
        models.RandomForest_SLearner(),
        models.RandomForest_TLearner(),
        models.DecisionTree_SLearner(),
        models.DecisionTree_TLearner(),
        models.MLP_SLearner(num_features = len(X_col) + 1, epochs = 50),
        models.MLP_TLearner(num_features = len(X_col), epochs = 50)
    ]

    with open('./data/results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["model", "PAPE", "n_treated_no_budget", "PAPE_budget"])

    for model in model_list:
        model.fit(X_train, T_train, Y_train)
        HTE_pred = model.predict(X_test)

        # get ITR
        ITR = HTE_pred > 0

        # get ITR where we only treat 1/2 of pop.
        num_treated = int(X_test.shape[0]/2)
        ITR_budget = get_budget_ITR(HTE_pred, num_treated)

        pape = PAPE(Y_test, T_test, ITR)
        pape_budget = PAPE(Y_test, T_test, ITR_budget)


        print(type(model).__name__, pape, pape_budget)
        with open('./data/results.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([type(model).__name__, pape, sum(ITR), pape_budget])
