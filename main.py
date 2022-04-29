from folktables import ACSDataSource, ACSIncome

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

from metrics import get_stats, get_col
import untreated
import postprocessed
import expgrad

# Train
train_state = "CA"
train_year = "2018"

data_source = ACSDataSource(survey_year=train_year, horizon='1-Year', survey='person')
acs_data = data_source.get_data(states=[train_state], download=True)

features, label, group = ACSIncome.df_to_numpy(acs_data)
X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(
    features, label, group, test_size=0.2, random_state=0)

untreated_model = untreated.train_model(X_train, y_train)
models = {
    "untreated": untreated_model,
    "postprocessed": postprocessed.train_model(untreated_model, X_train, group_train, y_train),
    "expgrad": expgrad.train_model(X_train, group_train, y_train)
}

# Test
for name, model in models.items():
    if name != "postprocessed":
        pi = permutation_importance(model, X_train, y_train, scoring="accuracy")
        importances_df = pd.DataFrame([pi.importances_mean], columns=["importance_mean_{}".format(i) for i in range(10)])
        importances_df.to_csv("metrics/{}_{}_{}_importance.csv".format(train_state, train_year, name), index=False)

    rows = []
    for year in ["2018"]:
        data_source = ACSDataSource(survey_year=year, horizon='1-Year', survey='person')
        for state in ["CA", "WA", "KY", "NY"]:
            if state == train_state:
                features, label, group = X_test, y_test, group_test
            else:
                acs_data = data_source.get_data(states=[state], download=True)
                features, label, group = ACSIncome.df_to_numpy(acs_data)

            if name == "postprocessed":
                stats = get_stats(model, True, state, year, features, group, label)
            else:
                stats = get_stats(model, False, state, year, features, group, label)

            rows.append(stats)

    df = pd.DataFrame(rows, columns=get_col())
    df.to_csv("metrics/{}_{}_{}.csv".format(train_state, train_year, name), index=False)
