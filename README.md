# Understanding Robust and Stable Fairness in a Classification Task
This is the repo for CS475/675 Machine Learning final project by:
- Alvin Kim (akim128)
- Kristen Nixon (knixon2)
- S.S. Diwangkara (ssangdi1)
- Sonia Jindal (sjindal1)

## Dependencies
- folktables
- fairlearn

## How to run
To run it, simply install the dependencies and run `main.py`. This will populate the `metrics` directory with CSV files that tells you about the model.

## About
This is the code that we use to produce the fairness metrics results for Gradient Boosted Classifier models with various fairness intervention method applied (plus the untreated one).

The file name of the CSV tells you about how the model is trained: `CA_2018_expgrad.csv` means that the model is trained on the 2018 Californian population using the expgrad fairness intervention method.

The content of the file tells you how the model will perform on a test data. 
- The 1st column tells you the state which the model is tested in
- The 2nd column tells you on what year the model is tested in
- The 3rd column tells you the accuracy score
- The 4th-12th column tells you the selection rate of a given racial group, this is used to compute demographic parity (e.g. black-white DP = selection_rate_1 - selection_rate_2)
- The 13th-21st column tells you the true positive rate of a given racial group, this is used to compute equality of opportunity (e.g. black-white EoO = true_positive_rate_1 - true_positive_rate_2)

Note that the results aren't complete since we faced some problem when training a postprocessed model with 0 positive / negative result for a given racial group: AL, FL, HI, ID, IL, KS, KY, ME, MN, MS, MO, MT, NH, NM, NY, NC, ND, OK, PA, RI, TN, UT, VT, VA, WV, WY

Aside from the performance results, there's also the feature importance metric using the `permutation_importance` function from `sklearn.inspection`, which is written to the corresponding `*_importance.csv` file.

## Visualization 
The notebook visualization is done on 3 files:
- `data_analysis.py` is used to compare distributions of different features in the data between states
- `result_figs.py` is used to plot feature importance and model accuracy w.r.t. fairness metrics
- `dsitribution_similarity.py` is used to plot the correlation analysis using chi-squared