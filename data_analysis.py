#%% this file compares distributions of data between states, used to make the "fraction of population by race" figure

# from folktables import ACSDataSource, ACSIncome
import pickle
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots

# data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
# acs_data = data_source.get_data(download=True)
# features, label, group = ACSIncome.df_to_numpy(acs_data)

#pickle out data
# outfile = open("all_states_acs_data", "wb")
# pickle.dump(acs_data, outfile)
# outfile.close()

#pickle in
infile = open("all_states_acs_data", "rb")
input_data = pickle.load(infile)
infile.close()

data = input_data[['ST','AGEP','COW','SCHL','MAR','OCCP','POBP','RELP','WKHP','SEX','RAC1P','PINCP']] #get only the variables for income prediction task
data['target'] = np.where(data['PINCP'] > 50000, True, False)
data['ST'] = data['ST'].astype(str).str.zfill(2)

#replace numbers with words so plots are interpretable
data['COW'] = data['COW'].replace({1:'private for-profit company', 2:'private not-for-profit',3:'local government',
                   4:'state government',5:'federal government',6:'self-employed in not incorporated business',
                   7:'self-employed in incorporated business',8: 'working without pay in family business',9:'unemployed'})
data['RAC1P'] = data['RAC1P'] .replace({1:'white', 2:'black',3:'american indian',
                   4:'alaska native',5:'native',6:'asian',
                   7:'pacific islander',8: 'other',9:'2+ races'})
data['SEX'] = data['SEX'] .replace({1:'male', 2:'female'})
data['SCHL'] = data['SCHL'] .replace({16:"high school", 20:"associate's degree", 21:"bachelor's degree"})
data['MAR'] = data['MAR'] .replace({1:'married', 2:'widowed',3:'divorced',
                   4:'separated',5:'never married'})


#%%
#plot each state separately

FIPS_lookup = pd.read_csv('state_lookup_K.csv')
FIPS_lookup['FIPS'] = FIPS_lookup['FIPS'].astype(str).str.zfill(2)
FIPS_lookup = FIPS_lookup.set_index('FIPS')
FIPS_lookup = FIPS_lookup[~FIPS_lookup.index.duplicated(keep='first')]


def place_from_FIPS(fips):
    state = FIPS_lookup.loc[fips, 'State']
    return state

state_regions_dict = {'Northeast':['09','23','25','33','44','50','34','36','42','24'],
                      'Midwest':['18','17','26','39','55','19','31','20','38','27','46','29'],
                      'South':['10','12','13','37','45','51','54','01','21','28','47','05','22','40','48'], # excluded MD ('24'), put in northeast
                      'West': ['04', '08', '16', '35', '30', '49', '32', '56', '02', '06', '15', '41', '53','72',],
                      }
# %%
#plot all states
var_to_plot = 'MAR' #which variable to plot
data = data.sort_values(by=var_to_plot) #this makes order of categories displayed the same on all subplots


subp_locs = [(1, 1), (1, 2),(1, 3),(1, 4),(2, 1),(2, 2),(2, 3),(2, 4),(3, 1),(3, 2),(3, 3),(3, 4),
(4, 1),(4, 2),(4, 3),(4, 4)]

for region in state_regions_dict:
    subp_counter = 0
    if region == 'Northeast':
        rows=3
    elif region == 'Midwest':
        rows=3
    else:
        rows=4

    fig = make_subplots(rows=rows, cols=4,
                    subplot_titles=tuple([place_from_FIPS(i) for i in state_regions_dict[region]]),shared_xaxes=True,shared_yaxes=True)


    for fips in state_regions_dict[region]:

        row = subp_locs[subp_counter][0]
        col = subp_locs[subp_counter][1]
        one_state = data[data['ST']==fips]

        fig.add_trace(go.Histogram(x=one_state[var_to_plot],showlegend=False,histnorm='percent'),row=row, col=col)
        fig.update_xaxes(tickangle=45)
        subp_counter += 1
    fig.update_layout(title=region + ': ' + var_to_plot)

    fig.show()

#%%
#summarize some of cats for a variable- one plot per region, group all states in a region into the same plot
var_to_plot = 'RAC1P'
cats_to_plot = ['other', '2+ races', 'asian','american indian'] #["high school","associate's degree","bachelor's degree"]


#get fractions of pop for each of cats to plot in one df
frac_plot = pd.DataFrame()
for fips in data['ST'].unique():
    one_state = data[data['ST'] == fips]
    counts = one_state[var_to_plot].value_counts()  #count how many occurances of each cat
    total = counts.sum() #get total number of records
    frac_plot[fips] = counts.loc[cats_to_plot]/total #get fraction for cats we care about



#need to set up subplots differently bc of the way bar plot object works
# import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools

traces_dict ={}
subp_counter = 0
colors=['red','blue','purple','green','orange','brown']
legend = True  # so legend is not repeated 4 times
for region in state_regions_dict:
    plot_states = frac_plot[state_regions_dict[region]]  # get only the states in this region
    plot_states.columns = [place_from_FIPS(i) for i in plot_states.columns] # rename columns from fips to state names
    cats_trace_list = []
    for count, cat in enumerate(cats_to_plot):
        trace1 = go.Bar(name=cat, x=plot_states.columns, y=plot_states.loc[cat, :],marker_color=colors[count],showlegend=legend)
        cats_trace_list.append(trace1)
    traces_dict.update({region: cats_trace_list})
    legend = False



fig = tools.make_subplots(rows=2, cols=2,shared_yaxes=True, subplot_titles=tuple(['Northeast','Midwest','South','West']))
subp_locs = [(1, 1),(1, 2),(2, 1),(2, 2)]
subp_counter = 0
for region in state_regions_dict:
    for i in range(0,len(cats_to_plot)):
        row = subp_locs[subp_counter][0]
        col = subp_locs[subp_counter][1]
        fig.append_trace(traces_dict[region][i], row,col)
    subp_counter += 1
fig.update_yaxes(range=[0, frac_plot.max().max()])
fig.update_xaxes(tickangle=45)

fig.update_layout(title=var_to_plot)
fig.show()

#%%
#plot grouped bar for selected states only
var_to_plot='RAC1P'
cats_to_plot = ["white","black"]
data_plot = data.set_index('ST').loc[['06','53','36','21'],:].reset_index()
frac_plot = pd.DataFrame()
for fips in data_plot['ST'].unique():
    one_state = data[data['ST'] == fips]
    counts = one_state[var_to_plot].value_counts()  #count how many occurances of each cat
    total = counts.sum() #get total number of records
    frac_plot[fips] = counts.loc[cats_to_plot]/total #get fraction for cats we care about
frac_plot.columns = [place_from_FIPS(i) for i in frac_plot.columns]

fig = go.Figure(data=[
    go.Bar(name='white', x=frac_plot.columns, y=frac_plot.loc['white',:]),
    go.Bar(name='black', x=frac_plot.columns, y=frac_plot.loc['black',:]),
])
# Change the bar mode
fig.update_layout(barmode='group',font=dict(
        size=18))
fig.show()

#%% get difference in white/black population
x=frac_plot.T
x['diff'] = x['white'] - x['black']
