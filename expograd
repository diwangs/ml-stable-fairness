from folktables import ACSDataSource, ACSEmployment

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from fairlearn.metrics import equalized_odds_difference, demographic_parity_difference
from metrics import equality_of_opportunity_difference

data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
acs_data = data_source.get_data(states=["CA"], download=True)
features, label, group = ACSEmployment.df_to_numpy(acs_data)

X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(
    features, label, group, test_size=0.2, random_state=0)

###### Your favorite learning algorithm here #####
model = make_pipeline(StandardScaler(), LogisticRegression())
model.fit(X_train, y_train)
y_hat = model.predict(X_test)

print("Untreated model:")
print("{:.5f} -> Accuracy".format(accuracy_score(y_test, y_hat)))
print("{:.5f} -> Maximum EoO disparity".format(equality_of_opportunity_difference(y_test, y_hat, sensitive_features=group_test)))
print("{:.5f} -> Equalized Odds disparity".format(equalized_odds_difference(y_test, y_hat, sensitive_features=group_test)))
black_white_yhat = y_hat[(group_test == 2) | (group_test == 1)]
black_white_ytest = y_test[(group_test == 2) | (group_test == 1)]
black_white_group = group_test[(group_test == 2) | (group_test == 1)]
print("{:.5f} -> EoO disparity between blacks and whites".format(equality_of_opportunity_difference(black_white_ytest, black_white_yhat, sensitive_features=black_white_group)))
print("{:.5f} -> Equalized Odds disparity between blacks and whites".format(equalized_odds_difference(black_white_ytest, black_white_yhat, sensitive_features=black_white_group)))
print()



expgrad = ExponentiatedGradient(
    LogisticRegression(solver='liblinear', fit_intercept=True),
    constraints=DemographicParity(),
    eps=0.01,
    nu=1e-6,
    max_iter = 10)


expgrad.fit(
    X_train,
    y_train,
    sensitive_features=group_train)

print("??????????????????????????????????????????????????????????????????????????")

yf_hat = expgrad.predict(X_test)


print("Treated model:")
print("{:.5f} -> Accuracy".format(accuracy_score(y_test, yf_hat)))
print("{:.5f} -> Maximum EoO disparity".format(equality_of_opportunity_difference(y_test, yf_hat, sensitive_features=group_test)))
print("{:.5f} -> Equalized Odds disparity".format(equalized_odds_difference(y_test, yf_hat, sensitive_features=group_test)))
black_white_yfhat = yf_hat[(group_test == 2) | (group_test == 1)]
print("{:.5f} -> EoO disparity between blacks and whites".format(equality_of_opportunity_difference(black_white_ytest, black_white_yfhat, sensitive_features=black_white_group)))
print("{:.5f} -> Equalized Odds disparity between blacks and whites".format(equalized_odds_difference(black_white_ytest, black_white_yfhat, sensitive_features=black_white_group)))
