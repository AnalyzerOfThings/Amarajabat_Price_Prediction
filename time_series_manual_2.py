import pandas as pd
#import model_functions
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
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.api import OLS
from pmdarima import auto_arima
import warnings
warnings.filterwarnings('ignore')

data = pd.read_excel('amarajabat.xlsx')
data['close_shifted_5'] = data['close'].shift(-5)
data = data.dropna()
data['return'] = (data['close_shifted_5']/data['close'])-1
data['shock'] = data['return'] - data['return'].mean()
data['shock_shifted_5'] = data['shock'].shift(5)
data['shock_shifted_6'] = data['shock'].shift(6)
data = data.dropna().reset_index(drop=True)

plot_acf(data['return'])
plt.show()
plot_pacf(data['return'])
plt.show()

features = ['shock_shifted_5','shock_shifted_6']

x = data[features]
y = data['return']
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False)

arma_14 = LinearRegression()
arma_14.fit(x_train, y_train)
y_pred = arma_14.predict(x)

data['arma_14_pred'] = y_pred + data['return'].mean() 

print('MSE: ',mean_squared_error(data['return'].iloc[-550:],
                         data['arma_14_pred'].iloc[-550:]))
print('R2: ',r2_score(data['return'], data['arma_14_pred']))
sns.scatterplot(data['return'])
sns.scatterplot(data['arma_14_pred'])
plt.show()
sns.scatterplot(x=data['return'], y=data['arma_14_pred'])
plt.xlabel('Return')
plt.ylabel('Predictions')
plt.show()

df = data[['return', 'arma_14_pred']].iloc[len(x_train):].reset_index(
    drop=True)
c=0
for i in range(len(df)):
    if df['return'].iloc[i] < 0 and df['arma_14_pred'].iloc[i] < 0:
        c+=1
    elif df['return'].iloc[i] > 0 and df['arma_14_pred'].iloc[i] > 0:
        c+=1
print('Accuracy: ', c/len(df))

x_train_ = statsmodels.api.add_constant(x_train)
model = OLS(y_train,x_train_)
res = model.fit()
print(res.summary())