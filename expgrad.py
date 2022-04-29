from sklearn.ensemble import GradientBoostingClassifier
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

def train_model(X, group, y):
    expgrad = ExponentiatedGradient(
        GradientBoostingClassifier(),
        constraints=DemographicParity(),
        max_iter = 5)

    expgrad.fit(
        X,
        y,
        sensitive_features=group)

    return expgrad