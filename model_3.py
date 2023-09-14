import pandas as pd
import statsmodels.api as sm
import model_functions
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import mean_absolute_error
from ta.trend import macd_diff, macd

################ LOADING FILE

path = 'amarajabat.xlsx'
data = pd.read_excel(path)


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
model_functions.add_label_numperiods_qlearn(data, 'days',20)
data['reco'] = model_functions.get_reco(data['days'],20)

validate_data = data[1600:]

############### BUILDING MODEL

features = ['macd','macd_diff','change_lag_1','change_lag_2','change_lag_5']
train_test_data = data[:1600]

x = train_test_data[features]
y = train_test_data.days

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=1000, shuffle=False)
train_data = pd.concat([x_train, y_train],axis=1) # USED FOR CALCULATING T STAT


model = LinearRegression()
model.fit(x_train, y_train)

preds1 = pd.Series(model.predict(x_test)) # PREDS FOR TESTING DATA

preds2 = pd.Series(model.predict(x_train)) # PREDS FOR TRAINING DATA

coef = model.coef_
variables = list(model.feature_names_in_)
n = len(coef)
t = []
p = []
for i in range(n):
    t_stat = model_functions.t_stat(coef[i], variables[i], 'days', train_data, preds2)
    p_val = model_functions.p_val(t_stat, len(x_train)-n-1)
    t.append(t_stat)
    p.append(p_val)

#####################

x_train_ = sm.add_constant(x_train)
model_sm = sm.OLS(y_train,x_train_)
res = model_sm.fit()
print('INITIAL')
print(res.summary())

condition_number = res.condition_number # ONLY THIS INDICATES SEVERE MULTICOLLINEARITY
corr = model_functions.corr(data, features)
vif = model_functions.vif(features, train_data)

init_test_r2 = model_functions.r2(y_train, preds2)
init_train_r2 = model_functions.r2(y_test, preds1)

del(model_sm)

############## FORWARD SELECTION

s = SequentialFeatureSelector(model, n_features_to_select='auto', tol=0.01, direction='forward', scoring='r2' )
s.fit(x_train, y_train)
final_feat = list(s.get_feature_names_out())
del(s)

############# MODEL USING CHOSEN FEATURES

x = train_test_data[final_feat]
y = train_test_data.days
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=1000, shuffle = False)
final_model = LinearRegression()
final_model.fit(x_train, y_train)
final_preds1 = pd.Series(final_model.predict(x_test)) # preds for test data
final_preds2 = pd.Series(final_model.predict(x_train)) # preds for train data


x_train_ = sm.add_constant(x_train)
model_sm = sm.OLS(y_train,x_train_)
res = model_sm.fit()
print('FINAL')
print(res.summary())

condition_number_ = res.condition_number 
corr_ = model_functions.corr(data, features)
final_test_r2 = model_functions.r2(y_test, final_preds1) 
final_train_r2 = model_functions.r2(y_train, final_preds2)
del(model_sm)

coef = final_model.coef_
variables = list(final_model.feature_names_in_)
n = len(coef)
t_ = []
p_ = []
for i in range(n):
    t_stat = model_functions.t_stat(coef[i], variables[i], 'days', train_data, final_preds2)
    p_val = model_functions.p_val(t_stat, len(x_train)-n-1)
    t_.append(t_stat)
    p_.append(p_val)
    
###################### PREDICT VALIDATION DATA

final_preds = final_model.predict(validate_data[final_feat])
final_recos = model_functions.get_reco(final_preds, 20)
mae = mean_absolute_error(validate_data.days, final_preds)

#######################

all_preds = list(preds1) + list(preds2) + list(final_preds)
ax = sns.scatterplot(x = all_preds, y = data['days'])
ax.set(xlabel = 'preds')
plt.show()

actual = sns.lineplot(x = data['date'], y = data['days'])
preds = sns.lineplot(x = data['date'], y = all_preds)
ax.set(ylabel = 'days', xlabel = 'date')
plt.show()

ax = sns.lineplot(x = data['date'], y =data['close'])
ax.set(ylabel = 'close')
plt.show()