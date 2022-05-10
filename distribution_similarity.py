#%% this file calculates the chi-square similarity metrics and completes the correlation analysis

import pickle
import pandas as pd
from scipy.stats import chi2_contingency
import plotly.graph_objects as go

#open and read data file
infile = open("all_states_acs_data", "rb")
input_data = pickle.load(infile)
infile.close()

#%%
#some data cleaning
data = input_data[['ST','RAC1P']] #get only the variables for income prediction task
data['ST'] = data['ST'].astype(str).str.zfill(2)
#data = data.loc[data['RAC1P'].isin([1,2])] ###only including white and black populations. Change if needed
data['RAC1P'] = data['RAC1P'].replace({1:'white', 2:'black',3:'american indian',
                   4:'alaska native',5:'native',6:'asian',
                   7:'pacific islander',8: 'other',9:'2+ races'})

#make dictionaries of FIPS codes and obbreviated state names
FIPS_lookup = pd.read_csv('state_lookup_K.csv')
FIPS_lookup['FIPS'] = FIPS_lookup['FIPS'].astype(str).str.zfill(2)
FIPS_lookup = FIPS_lookup.set_index('FIPS')
FIPS_lookup = FIPS_lookup[~FIPS_lookup.index.duplicated(keep='first')]
FIPS_dict_full = FIPS_lookup['State'].to_dict()
abbr_lookup = FIPS_lookup.set_index('Abbrev')
abbr_dict = abbr_lookup['State'].to_dict()

#in data, replace fips codes w/state names
data['ST'] = data['ST'].map(FIPS_dict_full)

#%%
#import csvs with fairness metrics for each train/test combo
states_to_use = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI','ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'MA', 'ME', 'MD', 'MI',
       'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC','ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
       'VT', 'VA', 'WA', 'WV', 'WI', 'WY']

untreated_dict = {}
for state in states_to_use:
    untreated = pd.read_csv("metrics/"+state+"_2018_untreated.csv", header=0,
                          usecols =["state","accuracy","selection_rate_1","selection_rate_2","true_pos_rate_1","true_pos_rate_2"])
    untreated['state'] = untreated['state'].map(abbr_dict)
    untreated['dem_parity'] = untreated['selection_rate_1'] - untreated['selection_rate_2']
    untreated['eq_opp'] = untreated['true_pos_rate_1'] - untreated['true_pos_rate_2']
    untreated_dict.update({abbr_dict[state]: untreated})


expgrad_dict = {}
for state in states_to_use:
    expgrad = pd.read_csv("metrics/"+state+"_2018_expgrad.csv", header=0,
                          usecols =["state","accuracy","selection_rate_1","selection_rate_2","true_pos_rate_1","true_pos_rate_2"])
    expgrad['state'] = expgrad['state'].map(abbr_dict)
    expgrad['dem_parity'] = expgrad['selection_rate_1'] - expgrad['selection_rate_2']
    expgrad['eq_opp'] = expgrad['true_pos_rate_1'] - expgrad['true_pos_rate_2']
    expgrad_dict.update({abbr_dict[state]: expgrad})

states_to_use_pp = ['AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'GA', 'IN', 'IA', 'LA', 'MA', 'MD', 'MI',
 'NE', 'NV', 'NJ', 'OH', 'OR', 'SC', 'SD', 'TX', 'WA', 'WI'] #had issues running fairness intervention/calculating metrics for Pennsylvania, Vermont, drop those

postproc_dict = {}
for state in states_to_use_pp:
    postproc = pd.read_csv("metrics/"+state+"_2018_postprocessed.csv", header=0,
                          usecols =["state","accuracy","selection_rate_1","selection_rate_2","true_pos_rate_1","true_pos_rate_2"])
    postproc['state'] = postproc['state'].map(abbr_dict)
    postproc['dem_parity'] = postproc['selection_rate_1'] - postproc['selection_rate_2']
    postproc['eq_opp'] = postproc['true_pos_rate_1'] - postproc['true_pos_rate_2']
    postproc_dict.update({abbr_dict[state]: postproc})

#%%
#plot plotly - so you can see which point is which state

import plotly.express as px
x = expgrad.sort_values(by='chi_value')
indexNames =expgrad[(expgrad['chi_value'] > 2500)].index
x = expgrad.drop(indexNames , inplace=False)

fig = px.scatter(expgrad, x='chi_value', y='dem_parity',hover_data=['state'], trendline="ols",title='VA - ExpGrad')
fig.show()

#%%
#do chi sq on all races - need to make an other category for small cats
#chi squared values should always be over 5
#check this

# exp_values_df = pd.DataFrame()
# for st in CA_expgrad["state"]:
#     states = ['California', 'Puerto Rico']
#     state_data = data.loc[data['ST'].isin(states)]
#     contingency_table = pd.crosstab(state_data['ST'], state_data['RAC1P']).T
#     chi2_value, p_value, dof, exp_values = chi2_contingency(contingency_table)
#     exp_values_df1 = pd.DataFrame(exp_values, index=contingency_table.index)
#     exp_values_df = pd.concat([exp_values_df, exp_values_df1]) #to check smallest exp value - needs to be greater than 5


#based on too low of exp values, move alaska native, pacific islander, native into other category
data = data.replace('alaska native', 'other')
data = data.replace('pacific islander', 'other')
data_with_other = data.replace('native', 'other')

#%%
def get_chi_sq(state,data,expgrad): #for each test state, get chi sq values between the train state and that test state
    chi = []
    pvals = []
    for st in expgrad["state"]: #loop through all test state
        states = [state, st]
        state_data = data.loc[data['ST'].isin(states)]
        contingency_table = pd.crosstab(state_data['ST'], state_data['RAC1P']).T
        chi2_value, p_value, dof, exp_values = chi2_contingency(contingency_table)
        chi.append(chi2_value)
        pvals.append(p_value)

    expgrad['chi_value'] = chi
    expgrad['pvalue'] = pvals
    return expgrad

#create new dict with chisq values included
untreated_chisq_dict = {}
for state in untreated_dict.keys():
    untreated_chisq = get_chi_sq(state, data_with_other, untreated_dict[state])
    untreated_chisq_dict.update({state: untreated_chisq})

expgrad_chisq_dict = {}
for state in expgrad_dict.keys():
    expgrad_chisq = get_chi_sq(state, data_with_other, expgrad_dict[state])
    expgrad_chisq_dict.update({state:expgrad_chisq})

postproc_chisq_dict = {}
for state in postproc_dict.keys():
    postproc_chisq = get_chi_sq(state, data_with_other, postproc_dict[state])
    postproc_chisq_dict.update({state:postproc_chisq})

both_methods_dict = {'expgrad':expgrad_chisq_dict, 'postproc':postproc_chisq_dict}

# outfile = open("fairness_metrics_dict", "wb")
# pickle.dump(both_methods_dict, outfile)
# outfile.close()
# import pickle
# infile = open("fairness_metrics_dict", "rb")
# both_methods_dict1 = pickle.load(infile)
# infile.close()
#%%
#for now look at SD, NJ, CA, GA, MI

#try calculating a couple different types of correlation
metric='dem_parity'
methods = ['pearson','kendall','spearman']
corr = pd.DataFrame(index=expgrad_chisq_dict.keys(), columns=methods)
for state in expgrad_chisq_dict.keys():
    for method in methods:
        state_data = expgrad_chisq_dict[state]
        data_corr = state_data[['state', metric,'chi_value']].corr(method=method)
        corr.loc[state, method] = data_corr.iloc[0,1]

corr_sample = corr.loc[['California','New Jersey','Georgia','South Dakota','Michigan'],:]

#%%
#focus on spearman, run for expgrad and postprocess at same time
method = 'spearman'
corr = pd.DataFrame(index=expgrad_chisq_dict.keys(),columns=['dem_parity','eq_opp'])

for state in expgrad_chisq_dict.keys():
    for metric in ['dem_parity','eq_opp']:
        state_data = expgrad_chisq_dict[state]
        data_corr = state_data[['state', metric,'chi_value']].corr(method=method)
        corr.loc[state, metric] = data_corr.iloc[0,1]

# corr_sample = corr.loc[['California','New Jersey','Georgia','South Dakota','Michigan'],:]
corr.to_csv('expgrad_corr')

corr_pp = pd.DataFrame(index=postproc_chisq_dict.keys(),columns=['dem_parity','eq_opp'])
for state in postproc_chisq_dict.keys():
    for metric in ['dem_parity','eq_opp']:
        state_data = postproc_chisq_dict[state]
        data_corr = state_data[['state', metric,'chi_value']].corr(method=method)
        corr_pp.loc[state, metric] = data_corr.iloc[0,1]

corr_u = pd.DataFrame(index=untreated_chisq_dict.keys(),columns=['dem_parity','eq_opp'])
for state in untreated_chisq_dict.keys():
    for metric in ['dem_parity','eq_opp']:
        state_data = untreated_chisq_dict[state]
        data_corr = state_data[['state', metric,'chi_value']].corr(method=method)
        corr_u.loc[state, metric] = data_corr.iloc[0,1]

#%% make these correlations into a heatmap

#get diff btwn black and white pop, want to sort by this on the heatmap
bw_diff_list = []
for state in data['ST'].unique():
    one_state = data[data['ST']==state]
    value_counts = one_state.value_counts().reset_index().set_index('RAC1P').drop('ST',axis=1)
    white_perc = int(value_counts.loc['white'])/int(value_counts.sum())
    black_perc = int(value_counts.loc['black']) / int(value_counts.sum())
    bw_diff_list.append(white_perc - black_perc)

bw_diff = pd.DataFrame(index=data['ST'].unique())
bw_diff['bw_diff'] = bw_diff_list

corr_pp.columns = ['Postprocessed, Demographic Parity', 'Postprocessed, Equality of Opportunity']
corr.columns = ['ExpGrad, Demographic Parity', 'ExpGrad, Equality of Opportunity']
corr_u.columns = ['No Intervention, Demographic Parity', 'No Intervention, Equality of Opportunity']
all_corrs1 = pd.concat([corr_u,corr],axis=1)
all_corrs = pd.concat([all_corrs1,corr_pp],axis=1)

all_corrs_bwdiff = all_corrs.merge(bw_diff,left_index=True, right_index=True).sort_values(by='bw_diff')
all_corrs_bwdiff_plot = all_corrs_bwdiff.drop('bw_diff',axis=1)

fig = go.Figure(data=go.Heatmap(z=all_corrs_bwdiff_plot, y=all_corrs_bwdiff_plot.index,x=all_corrs_bwdiff_plot.columns,colorscale='rdbu'))
fig.update_xaxes(title='Fairness Intervention, Fairness Metric')
fig.update_yaxes(title='State that Model was Trained On')
fig.update_layout(title='Heatmap of Spearman Correlations Between Similarity of Race Distribution between Train and Test State and Resulting Fairness Metric')
fig.show()

#%%

#%%
fig = px.scatter(expgrad_chisq_dict['Wisconsin'], x='chi_value', y='dem_parity',hover_data=['state'], trendline="ols",title='CA - ExpGrad')
fig.update_xaxes(title='Chi-Square Value (higher is more dissimilar)')
fig.update_yaxes(title='Demographic Parity (higher is more unfair)')
fig.update_layout(title='Scatter Plot Between Similarity of Race Distribution between Wisconsin and Test State and Resulting Fairness Metric')
fig.show()

fig = px.scatter(CA_expgrad_chisq, x='chi_value', y='eq_odds',hover_data=['state'], trendline="ols",title='CA - ExpGrad')
fig.show()



#%%
