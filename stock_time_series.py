import pandas as pd
import model_functions
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.stattools import adfuller, acf, pacf
from scipy.stats import norm
from scipy import stats
import statsmodels.api
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import warnings
warnings.filterwarnings('ignore')

data = pd.read_excel('amarajabat.xlsx')
data['close_shifted'] = data['close'].shift()
data['volume'] = np.log(data['volume'])
data = data.dropna()
data['return'] = (data['close']/data['close_shifted'])-1
data = data.drop('close_shifted', axis=1)

for col in data.columns[1:]:
    data[col] =  StandardScaler().fit_transform(data[col][:,np.newaxis])

# Make note of all vars, their range and their dtypes

print(data.dtypes)
for col in list(data.columns):
    print(col,': (min, max) =  ', (min(data[col]),max(data[col])))
    
# Analyse target

target = 'return'
print(data[target].describe())
sns.displot(data[target])
plt.show()
print('Skewness: ', data[target].skew())
print('Kurtosis: ',data[target].kurt())

contn_vars = ['open', 'low', 'high', 'close', 'volume']

for var in contn_vars:
    ax = sns.scatterplot(data, x = var, y = target)
    plt.show()    

# Analyse features

corrmat = data.corr()
f, ax = plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.8, square=True)
plt.show()

k = 6
cols = corrmat.nlargest(k, target)[target].index
cm = np.corrcoef(data[cols].values.T)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
                 annot_kws={'size':10}, yticklabels=cols.values,
                 xticklabels=cols.values)
plt.show()

for col1 in cols:
    for col2 in cols:
        if col1!=col2:
            sns.scatterplot(data=data, x=col1, y=col2)
            plt.show()

# Missing Data

total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending
                                                                  = False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

# Remove col if percent > 15 

# Outliers

target_scaled = list(data['return']).copy()
target_scaled.sort()
lowrange = target_scaled[:10]
highrange = target_scaled[-10:]
print('Lower end of distribution: ')
print(lowrange)
print('Higher end of distribution: ')
print(highrange)

# Check if vars conform to statistical test assumptions

## Normality

for col in cols:
    sns.distplot(data[col], fit=norm)
    fig = plt.figure()
    res = stats.probplot(data[col], plot=plt)
    plt.title(col+' Probability Plot')
    plt.show()

for col in cols:
    sns.scatterplot(data=data, y=target, x=col)
    plt.show()

# Fitting Models

data = pd.read_excel('amarajabat.xlsx')
data['close_shifted'] = data['close'].shift()
data = data.dropna()
data['return'] = (data['close']/data['close_shifted'])-1
data['return'] = StandardScaler().fit_transform(
    data['return'][:,np.newaxis])
data = data.drop('close_shifted', axis=1)
print('ADF: ', adfuller(data['return'])[0])

plot_acf(data['return'])
plt.show()
plot_pacf(data['return'])
plt.show()

data['return_shifted'] = data['return'].shift()
data = data.dropna()

x = data['return_shifted']
y = data['return']
x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle = False,
                                                random_state=0)

model = LinearRegression()
model.fit(np.array(x_train).reshape(-1,1), y_train)
y_pred = model.predict(np.array(x_test).reshape(-1,1))

ar = ARIMA(y_train, order=(1,0,0))
res_ar = ar.fit()
pred_ar = res_ar.predict(start=1655, end=2206)
print(res_ar.summary())

ma = ARIMA(y_train, order=(0,0,1))
res_ma = ma.fit()
pred_ma = res_ma.predict(start = 1655, end = 2206)
print(res_ma.summary())

arma = ARIMA(y_train, order=(1,0,1))
res_arma = arma.fit()
pred_arma = res_arma.predict(start = 1655, end = 2206)
print(res_arma.summary())

arima = ARIMA(y_train, order=(1,1,1))
res_arima = arima.fit()
pred_arima = res_arima.predict(start = 1655, end = 2206)
print(res_arima.summary())

pred_lin = [np.nan] * (2205-552)
for i in y_pred:
    pred_lin.append(i)
    
data['pred_lin'] = pred_lin
data['pred_ar'] = pred_ar
data['pred_ma'] = pred_ma
data['pred_arma'] = pred_arma
data['pred_arima'] = pred_arima

for i in list(data.columns[-5:]):
    print(i,' ', mean_squared_error(data['return'].iloc[1655:], 
                                    data[i].iloc[1655:]))

sns.scatterplot(data['return'].iloc[1655:], label='return')
sns.scatterplot(data['pred_lin'].iloc[1655:], label='pred_lin')
sns.scatterplot(data['pred_ar'].iloc[1655:], label='pred_ar')
sns.scatterplot(data['pred_ma'].iloc[1655:], label='pred_ma')
sns.scatterplot(data['pred_arma'].iloc[1655:], label='pred_arma')
sns.scatterplot(data['pred_arima'].iloc[1655:], label='pred_arima')
plt.show()