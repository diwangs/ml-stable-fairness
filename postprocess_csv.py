from folktables import ACSDataSource, ACSEmployment, ACSIncome

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from metrics import equality_of_opportunity_difference, get_sr_metric_frame, get_tpr_metric_frame
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.metrics import equalized_odds_difference, demographic_parity_difference

data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
acs_data = data_source.get_data(states=["CA"], download=True)
wa_data = data_source.get_data(states=["WA"], download=True)
ky_data = data_source.get_data(states=["KY"], download=True)

features, label, group = ACSIncome.df_to_numpy(acs_data)
wa_features, wa_label, wa_group = ACSIncome.df_to_numpy(wa_data)
ky_features, ky_label, ky_group = ACSIncome.df_to_numpy(ky_data)

X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(
    features, label, group, test_size=0.2, random_state=0)

###### Your favorite learning algorithm here #####
model = make_pipeline(StandardScaler(), GradientBoostingClassifier())
model.fit(X_train, y_train)

def get_stats(model, treated, state, X, group, y):
    if treated:
        y_hat = model.predict(X, sensitive_features=group)
    else:
        y_hat = model.predict(X)
    acc = accuracy_score(y, y_hat)
    sr = get_sr_metric_frame(y, y_hat, sensitive_features=group).by_group["sr"].tolist()
    tpr = get_tpr_metric_frame(y, y_hat, sensitive_features=group).by_group["tpr"].tolist()
    row = {
        "state": state,
        "acc": acc,
        "selection_rate_1": sr[0],
        "selection_rate_2": sr[1],
        "selection_rate_3": sr[2],
        "selection_rate_4": sr[3],
        "selection_rate_5": sr[4],
        "selection_rate_6": sr[5],
        "selection_rate_7": sr[6],
        "selection_rate_8": sr[7],
        "selection_rate_9": sr[8],
        "true_pos_rate_1": tpr[0],
        "true_pos_rate_2": tpr[1],
        "true_pos_rate_3": tpr[2],
        "true_pos_rate_4": tpr[3],
        "true_pos_rate_5": tpr[4],
        "true_pos_rate_6": tpr[5],
        "true_pos_rate_7": tpr[6],
        "true_pos_rate_8": tpr[7],
        "true_pos_rate_9": tpr[8],
    }
    return row

col = ["state",
        "acc",
        "selection_rate_1",
        "selection_rate_2",
        "selection_rate_3",
        "selection_rate_4",
        "selection_rate_5",
        "selection_rate_6",
        "selection_rate_7",
        "selection_rate_8",
        "selection_rate_9",
        "true_pos_rate_1",
        "true_pos_rate_2",
        "true_pos_rate_3",
        "true_pos_rate_4",
        "true_pos_rate_5",
        "true_pos_rate_6",
        "true_pos_rate_7",
        "true_pos_rate_8",
        "true_pos_rate_9"
]

wa_untreated_stats = get_stats(model, False, "WA", wa_features, wa_group, wa_label)
ky_untreated_stats = get_stats(model, False, "KY", ky_features, ky_group, ky_label)
untreated_df = pd.DataFrame([wa_untreated_stats, ky_untreated_stats], columns=col)
untreated_df.to_csv("CA_untreated.csv", index=False)
