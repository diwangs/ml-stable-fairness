##%this file was used to plot accuracy of models vs fairness and look at feature importance

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

reg_fair = pd.read_csv('metrics/CA_2018_untreated.csv')
reg_imp = pd.read_csv('metrics/CA_2018_untreated_importance.csv')

expgrad_fair = pd.read_csv('metrics/CA_2018_expgrad.csv')
expgrad_imp = pd.read_csv('metrics/CA_2018_expgrad_importance.csv')

post_fair = pd.read_csv('metrics/CA_2018_postprocessed.csv')

#%%
#importance
reg_imp.columns = ['AGEP','COW','SCHL','MAR','OCCP', 'POBP','RELP', 'WKHP', 'SEX', 'RAC1P']
expgrad_imp.columns = ['AGEP','COW','SCHL','MAR','OCCP', 'POBP','RELP', 'WKHP', 'SEX', 'RAC1P']

imp = pd.concat([reg_imp, expgrad_imp]).T
imp.columns = ['No Intervention', 'ExpGrad']

imp= imp.reset_index()
imp['index'] = imp['index'] .replace({'AGEP':'Age', 'COW':'Class of Work','SCHL':'Level of Education',
                   'MAR':'Marital Status','OCCP':'Occupation','POBP':'Place of Birth','RELP':'Relationship to Person Filling Out Census',
                                      'WKHP':'Hours Worked','SEX':'Sex','RAC1P':'Race'})
imp = imp.set_index('index')

#%%
#fairness/accuracy
reg_fair['method'] = 'regular'
expgrad_fair['method'] = 'expgrad'
post_fair['method'] = 'post'
fair = imp = pd.concat([reg_fair, expgrad_fair,post_fair])

#calculate fairness metrics btwn black and white
fair['Demographic Parity-white and black people'] = fair['selection_rate_1'] - fair['selection_rate_2']
fair['Equalized Odds-black and white people'] = fair['true_pos_rate_1'] - fair['true_pos_rate_2']

# fair['state'] = data['state'] .replace({'CA':'California', 'WA':'Washington','KY':'Kentucky','NY':'New York'})

#%% plot - recreate figure from paper
subp_locs = [(1, 1),(1, 2),(2, 1),(2, 2), (3, 1),(3, 2),]
subp_counter=0
varname_to_plotname = dict({'regular': 'None', 'expgrad':'ExpGrad','post':'Postprocessing'})
fig = make_subplots(rows=3, cols=2)
for method in fair['method'].unique():
    data = fair[fair['method']==method]
    for metric in ['Demographic Parity-white and black people', 'Equalized Odds-black and white people']:
        row = subp_locs[subp_counter][0]
        col = subp_locs[subp_counter][1]
        fig.add_trace(go.Scatter(x=data['accuracy'], y=data[metric], mode='markers+text', text=data['state'],textposition="bottom center",textfont=dict(size=20),marker=dict(size=[20, 20, 20, 20],
                color=['blue', 'red','red','red'])),row=row,col=col)
        fig.update_yaxes(title=metric)
        fig.update_xaxes(title='Accuracy')
        subp_counter += 1
    # fig.update_yaxes(title = metric)
    # fig.update_xaxes(title='Accuracy')
    # fig.update_layout(title='Fairness Intervention: '+varname_to_plotname[method] + ', Fairness Metric: ' + metric)
fig.update_yaxes(range=[-.15, .3],title='Fairness')
fig.update_xaxes(range=[.70, .83],title='Accuracy')
fig.show()

#%% same figure - but 2x3 subplots instead of 3x2 - I think this is better because it's easier to compare fairness across different interventions
subp_locs = [(1, 1),(2, 1),(1, 2),(2, 2), (1, 3),(2, 3)]
subp_counter=0
varname_to_plotname = dict({'regular': 'None', 'expgrad':'ExpGrad','post':'Postprocessing'})
fig = make_subplots(rows=2, cols=3)
for method in fair['method'].unique():
    data = fair[fair['method']==method]
    for metric in ['Demographic Parity-white and black people', 'Equalized Odds-black and white people']:
        row = subp_locs[subp_counter][0]
        col = subp_locs[subp_counter][1]
        fig.add_trace(go.Scatter(x=data['accuracy'], y=data[metric], mode='markers+text', text=data['state'],textposition="middle left",textfont=dict(size=20),marker=dict(size=[20, 20, 20, 20],
                color=['blue', 'red','red','red'])),row=row,col=col)
        fig.update_yaxes(title=metric)
        fig.update_xaxes(title='Accuracy')
        subp_counter += 1
    # fig.update_yaxes(title = metric)
    # fig.update_xaxes(title='Accuracy')
    # fig.update_layout(title='Fairness Intervention: '+varname_to_plotname[method] + ', Fairness Metric: ' + metric)
fig.update_yaxes(range=[-.15, .3],title='Fairness')
fig.update_xaxes(range=[.70, .83],title='Accuracy')
fig.show()
