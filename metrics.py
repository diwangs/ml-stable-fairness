from fairlearn.metrics import true_positive_rate, selection_rate, MetricFrame
from sklearn.metrics import accuracy_score

"""
https://arxiv.org/pdf/1610.02413.pdf
Many fairness metrics:

- Demographic Parity -> P(Y_hat = 1 | A = 0) - P(Y_hat = 1 | A = 1) 
    - Multiclass -> find the largest differenceh
    - Only considers "positive" label
    - Limits the utility of A, what if there's a correlation between A and Y?
- Equality of Opportunity -> P(Y_hat = 1 | A = 0, Y = 1) - P(Y_hat = 1 | A = 1, Y = 1)
    - Still has the first problem
    - Also known as True Positive Rate Disparity
- Equalized Odds -> EoO but also check false positive rate (in addition to true positive)
    - Disparity = max of EoO and false positive rate disparity
"""

def get_stats(model, treated, state, year, X, group, y):
    if treated:
        y_hat = model.predict(X, sensitive_features=group)
    else:
        y_hat = model.predict(X)
    acc = accuracy_score(y, y_hat)
    sr = get_sr_metric_frame(y, y_hat, sensitive_features=group).by_group["sr"].tolist()
    tpr = get_tpr_metric_frame(y, y_hat, sensitive_features=group).by_group["tpr"].tolist()
    row = [state, year, acc, *sr, *tpr]
    return row

def get_col():
    return ["state", 
        "year",
        "accuracy",
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

def get_sr_metric_frame(
        y_true,
        y_pred,
        *,
        sensitive_features,
        sample_weight=None
    ):
    sw_dict = {'sample_weight': sample_weight}
    sp = {'sr': sw_dict}
    eo = MetricFrame(
        metrics={'sr': selection_rate},
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
        sample_params=sp)
    return eo

def get_tpr_metric_frame(
        y_true,
        y_pred,
        *,
        sensitive_features,
        sample_weight=None
    ):
    sw_dict = {'sample_weight': sample_weight}
    sp = {'tpr': sw_dict}
    eo = MetricFrame(
        metrics={'tpr': true_positive_rate},
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
        sample_params=sp)
    return eo

def equality_of_opportunity_difference(
        y_true,
        y_pred,
        *,
        sensitive_features,
        sample_weight=None,
        method='between_groups'):
    eo = get_tpr_metric_frame(y_true, y_pred, sensitive_features=sensitive_features, sample_weight=sample_weight)

    return eo.difference(method=method)["tpr"] # Accross all groups, not just black and white