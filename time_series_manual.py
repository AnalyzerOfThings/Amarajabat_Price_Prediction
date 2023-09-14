import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

data = pd.read_excel('amarajabat.xlsx')
data['close_shifted_5'] = data['close'].shift(-5)
data['volume'] = np.log(data['volume'])
data['return'] = (data['close']/data['close_shifted_5'])-1
data['return_shifted_1'] = data['return'].shift(1)
data['shock'] = data['return'].mean() - data['return']
data['shock_shifted_1'] = data['shock'].shift(1)
data['shock_shifted_2'] = data['shock'].shift(2)
data['shock_shifted_3'] = data['shock'].shift(3)
data['shock_shifted_4'] = data['shock'].shift(4)
data = data.dropna().reset_index(drop=True)

plot_acf(data['return'])
plt.show()
plot_pacf(data['return'])
plt.show()

features = ['shock_shifted_1', 'shock_shifted_2', 'shock_shifted_3',
            'shock_shifted_4', 'return_shifted_1']

x = data[features]
y = data['return']
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False)

arma_14 = LinearRegression()
arma_14.fit(x_train, y_train)
y_pred = arma_14.predict(x)


data['arma_14_pred'] = y_pred

print('MSE: ',mean_squared_error(data['return'].iloc[-550:],
                         data['arma_14_pred'].iloc[-550:]))
sns.scatterplot(data['return'])
sns.scatterplot(data['arma_14_pred'])
plt.show()


df = data[['return', 'arma_14_pred']].iloc[1657:].reset_index(drop=True)
c=0
for i in range(len(df)):
    if df['return'].iloc[i] < 0 and df['arma_14_pred'].iloc[i] < 0:
        c+=1
    elif df['return'].iloc[i] > 0 and df['arma_14_pred'].iloc[i] > 0:
        c+=1
print('Accuracy: ', c/len(df))