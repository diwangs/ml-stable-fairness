from folktables import ACSDataSource, ACSIncome

import warnings
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

from metrics import get_stats, get_col
import untreated
import postprocessed
import expgrad

# Train
train_years = ["2018"]
train_states = ["PA"]
test_years = ["2018"]
test_states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "MA", "ME", "MD", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]
# Some has postprocessing problem
# PA, VT self-test problem
# LA, MA, WY test problem

for train_year in train_years:
    for train_state in tqdm(train_states):
        data_source = ACSDataSource(survey_year=train_year, horizon='1-Year', survey='person')
        acs_data = data_source.get_data(states=[train_state], download=True)

        features, label, group = ACSIncome.df_to_numpy(acs_data)
        X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(
            features, label, group, test_size=0.2, random_state=0)

        models = {}
        untreated_model = untreated.train_model(X_train, y_train)
        models["untreated"] = untreated_model
        try:
            postprocessed_model = postprocessed.train_model(untreated_model, X_train, group_train, y_train)
            models["postprocessed"] = postprocessed_model
        except:
            print("{} has postprocessing error".format(train_state))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            models["expgrad"] = expgrad.train_model(X_train, group_train, y_train)
        
        # models = {
        #     "untreated": untreated_model,
        #     "postprocessed": postprocessed_model,
        #     "expgrad": expgrad_model
        # }

        # Test
        for name, model in models.items():
            if name != "postprocessed":
                pi = permutation_importance(model, X_train, y_train, scoring="accuracy")
                importances_df = pd.DataFrame([pi.importances_mean], columns=["importance_mean_{}".format(i) for i in range(10)])
                importances_df.to_csv("metrics/{}_{}_{}_importance.csv".format(train_state, train_year, name), index=False)

            rows = []
            for test_year in ["2018"]:
                data_source = ACSDataSource(survey_year=test_year, horizon='1-Year', survey='person')
                for test_state in tqdm(test_states):
                    if test_year == train_year and test_state == train_state:
                        features, label, group = X_test, y_test, group_test
                    else:
                        acs_data = data_source.get_data(states=[test_state], download=True)
                        features, label, group = ACSIncome.df_to_numpy(acs_data)

                    try:
                        if name == "postprocessed":
                            stats = get_stats(model, True, test_state, test_year, features, group, label)
                        else:
                            stats = get_stats(model, False, test_state, test_year, features, group, label)
                    except:
                        print("{}->{} has test error".format(train_state, test_state))
                        continue

                    rows.append(stats)

            df = pd.DataFrame(rows, columns=get_col())
            df.to_csv("metrics/{}_{}_{}.csv".format(train_state, train_year, name), index=False)
