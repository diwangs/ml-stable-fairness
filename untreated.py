from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import make_pipeline

def train_model(X, y):
    model = make_pipeline(StandardScaler(), GradientBoostingClassifier())
    model.fit(X, y)

    return model
