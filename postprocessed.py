from fairlearn.postprocessing import ThresholdOptimizer

def train_model(prev_model, X, group, y):
    postprocess = ThresholdOptimizer(
        estimator=prev_model, 
        constraints="equalized_odds",
        prefit=True,
        predict_method="auto")
    postprocess.fit(X, y, sensitive_features=group)

    return postprocess