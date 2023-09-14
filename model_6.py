import pandas as pd
import model_functions
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from ta.trend import macd_diff, macd

################ LOADING FILE

data = pd.read_excel('amarajabat.xlsx')


################ ADDING FEATURES TO DATAFRAME

data['macd'] = macd(data['close'], fillna = True)
data['macd_diff'] = macd_diff(data['close'], fillna= True)
data['change_lag_1'] = ((data['close'].shift(1)/data['close'])-1)
data['change_lag_2'] = ((data['close'].shift(2)/data['close'])-1)
data['change_lag_5'] = ((data['close'].shift(5)/data['close'])-1)

data['change_lag_1'] = data['change_lag_1'].fillna(data['change_lag_1'].mean())
data['change_lag_2'] = data['change_lag_2'].fillna(data['change_lag_2'].mean())
data['change_lag_5'] = data['change_lag_5'].fillna(data['change_lag_5'].mean())
data['return_'] = data['close'].diff().fillna(0)

for i in [20, 50, 100]:
    model_functions.add_label_numperiods_qlearn(data, 'days_'+str(i), i)
    data['reco_'+str(i)] = model_functions.get_reco(
        data['days_'+str(i)], max(data['days_'+str(i)]))
 
dataframes = [[0, 0, 0],[0, 0, 0], [0, 0, 0]]
targets = ['days_20', 'days_50', 'days_100']
features = ['macd','macd_diff','change_lag_1','change_lag_2','change_lag_5']
features.append('close')
x = data[features]
features.remove('close')
for i in range(len(dataframes)):
    y = data[targets[i]]
    x_train = x[:1000]
    y_train = y[:1000]
    x_test = x[1000:]
    y_test = y[1000:]
    
    dataframes[i][0] = pd.concat([x_train, y_train],axis=1) 
    x_test, x_validate, y_test, y_validate = train_test_split(x_test,
                                        y_test, train_size=0.5, shuffle=False)
    dataframes[i][1] = pd.concat([x_test, y_test], axis=1)
    dataframes[i][2] = pd.concat([x_validate, y_validate], axis=1)

for i in dataframes:
    for j in i:
        j.reset_index(inplace=True, drop=True)

model = LinearRegression()
dataframes = model_functions.get_dataframes(model, features, dataframes,
                                            targets)

'''
############## FORWARD SELECTION

s = SequentialFeatureSelector(model, n_features_to_select='auto', tol=0.01,
                              direction='forward', scoring='r2' )
s.fit(x_train, y_train)
final_feat = list(s.get_feature_names_out())
del(s)
'''


final_feat = ['macd', 'macd_diff']
dataframes = model_functions.get_dataframes(model, final_feat, dataframes,
                                            targets)
r2_mod_2 = []
r2_mod_5 = []
adj_r2_mod_2 = []
adj_r2_mod_5 = []
returns_mod_2 = []
returns_mod_5 = []
for i in range(len(dataframes)):
    for j in dataframes[i]:
        r2_mod_2.append(model_functions.r2(j[targets[i]], j['preds_mod_2']))
        r2_mod_5.append(model_functions.r2(j[targets[i]], j['preds_mod_5']))
        adj_r2_mod_2.append(model_functions.adj_r2(j[targets[i]],
                                                   j['preds_mod_2'],2))
        adj_r2_mod_5.append(model_functions.adj_r2(j[targets[i]],
                                                   j['preds_mod_5'],5))
        returns_mod_2.append(sum(model_functions.get_return(j,
                                                            j['reco_mod_2'])))
        returns_mod_5.append(sum(model_functions.get_return(j,
                                                            j['reco_mod_5'])))                    
        

        
names = ['train_20','test_20','validate_20','train_50','test_50','validate_50',
         'train_100','test_100','validate_100']

r2 = pd.DataFrame({'data':names, 'mod_2':r2_mod_2, 'mod_5':r2_mod_5})
adj_r2 = pd.DataFrame({'data':names, 'mod_2':adj_r2_mod_2,
                       'mod_5':adj_r2_mod_5})
returns = pd.DataFrame({'data':names, 'mod_2':returns_mod_2,
                        'mod_5':returns_mod_5})

max_20_data = pd.concat([dataframes[0][0], dataframes[0][1], dataframes[0][2]],
                        axis=0).reset_index(drop=True)
max_50_data = pd.concat([dataframes[1][0], dataframes[1][1], dataframes[1][2]],
                        axis=0).reset_index(drop=True)
max_100_data = pd.concat([dataframes[2][0], dataframes[2][1], dataframes[2][2]]
                         ,axis=0).reset_index(drop=True)

dataframes = [max_20_data, max_50_data, max_100_data]

# PREDS VS TIME PLOTS

plt.title('max_days = 20')
sns.lineplot(x = data['date'], y = max_20_data['preds_mod_2'], color='red')
sns.lineplot(x = data['date'], y = max_20_data['preds_mod_5'])
sns.lineplot(x = data['date'], y = max_20_data['days_20'], color='yellow')
plt.ylabel('days')
plt.show()

plt.title('max_days = 50')
sns.lineplot(x = data['date'], y = max_50_data['preds_mod_2'], color='red')
sns.lineplot(x = data['date'], y = max_50_data['preds_mod_5'])
sns.lineplot(x = data['date'], y = max_50_data['days_50'], color='yellow')
plt.ylabel('days')
plt.show()

plt.title('max_days = 100')
sns.lineplot(x = data['date'], y = max_100_data['preds_mod_2'], color='red')
sns.lineplot(x = data['date'], y = max_100_data['preds_mod_5'])
sns.lineplot(x = data['date'], y = max_100_data['days_100'], color='yellow')
plt.ylabel('days')
plt.show()

for col in features:
    sns.scatterplot(x = max_20_data[col], y = max_20_data['days_20'])
    plt.xlabel(col)
    plt.show()
    sns.scatterplot(x = max_50_data[col], y = max_50_data['days_50'])
    plt.xlabel(col)
    plt.show()
    sns.scatterplot(x = max_100_data[col], y = max_100_data['days_100'])
    plt.xlabel(col)
    plt.show()
        

# ODDS OF MAKING PROFIT

odds_profit_max_20 = model_functions.odds_profit(data, data['reco_20'], 
                                                 'days_20', 20)
odds_profit_max_50 = model_functions.odds_profit(data, data['reco_50'], 
                                                 'days_50', 50)
odds_profit_max_100 = model_functions.odds_profit(data, data['reco_100'], 
                                                 'days_100', 100)

'''
for each of the above lists, index is the corresponding number of days
odds profit is higher the higher the number of days as per above function.

'''
sns.lineplot(odds_profit_max_20)
sns.lineplot(odds_profit_max_50, color='red')
sns.lineplot(odds_profit_max_100, color='yellow')
plt.title('Odds of Profit')
plt.xlabel('days')
plt.ylabel('odds')
plt.show()

sns.histplot(data['days_20'])
plt.show()
sns.histplot(data['days_50'])
plt.show()
sns.histplot(data['days_100'])
plt.show()