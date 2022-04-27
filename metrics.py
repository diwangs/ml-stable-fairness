from fairlearn.metrics import true_positive_rate, MetricFrame

"""
https://arxiv.org/pdf/1610.02413.pdf
Many fairness metrics:

- Demographic Parity -> P(Y_hat = 1 | A = 0) - P(Y_hat = 1 | A = 1) 
    - Multiclass -> find the largest difference
    - Only considers "positive" label
    - Limits the utility of A, what if there's a correlation between A and Y?
- Equality of Opportunity -> P(Y_hat = 1 | A = 0, Y = 1) - P(Y_hat = 1 | A = 1, Y = 1)
    - Still has the first problem
    - Also known as True Positive Rate Disparity
- Equalized Odds -> EoO but also check false positive rate (in addition to true positive)
    - Disparity = max of EoO and false positive rate disparity
"""

def equality_of_opportunity_difference(
        y_true,
        y_pred,
        *,
        sensitive_features,
        method='between_groups',
        sample_weight=None):
    sw_dict = {'sample_weight': sample_weight}
    sp = {'tpr': sw_dict}
    eo = MetricFrame(
        metrics={'tpr': true_positive_rate},
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
        sample_params=sp)

    return eo.difference(method=method)["tpr"] # Accross all groups, not just black and white