import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from statsmodels.tsa.stattools import adfuller, acf, pacf
import warnings

warnings.filterwarnings('ignore')


# MODEL : SHOCK_PREDICTED + MEAN = MEAN + K1*SHOCK_SHIFTED_5 + 
#                                         K2*SHOCK_SHIFTED_6
# SHOCK_PREDICTED + MEAN = RETURN_PREDICTED


data = pd.read_excel('amarajabat.xlsx')
data['return'] = ((data['close'].shift(-5)/data['close'])-1)*100
data['shock'] = data['return'] - data['return'].mean()
'''
for i in range(len(data['return'])):
    if data['return'].iloc[i] > 0:
        data['return'].iloc[i] = 1
    else:
        data['return'].iloc[i] = 0

for i in range(len(data['shock'])):
    if data['shock'].iloc[i] > 0:
        data['shock'].iloc[i] = 1
    else:
        data['shock'].iloc[i] = 0
'''

data['shock_shifted_1'] = data['shock'].shift(1)
data['shock_shifted_2'] = data['shock'].shift(2)
data['shock_shifted_3'] = data['shock'].shift(3)
data['shock_shifted_4'] = data['shock'].shift(4)
data['shock_shifted_5'] = data['shock'].shift(5)
data['shock_shifted_6'] = data['shock'].shift(6)

data = data.dropna().reset_index(drop=True)

models = [LinearRegression(), DecisionTreeRegressor(),
          RandomForestRegressor()]
for mod in models:
    model = mod
    x = data[['shock_shifted_5','shock_shifted_6']]
    y = data['shock']
    x_train, x_test, y_train, y_test = train_test_split(x,y,shuffle=False,
                                                        random_state=0)
    model.fit(x_train, y_train)
    data['shock_pred'] = model.predict(x)

    df = data[['shock', 'shock_pred']].iloc[len(x_train):].reset_index(
        drop=True)
    c=0
    for i in range(len(df)):
        if df['shock'].iloc[i] > 0 and df['shock_pred'].iloc[i] > 0:
            c+=1
        elif df['shock'].iloc[i] < 0 and df['shock_pred'].iloc[i] < 0:
            c+=1
    print('########')
    print(mod)
    print('Accuracy: ', c/len(df))
    print('MSE: ', mean_squared_error(data['shock'].iloc[len(x_train):], 
                                      data['shock_pred'].iloc[len(x_train):]))
    print('R2: ', r2_score(data['shock'].iloc[len(x_train):], 
                                      data['shock_pred'].iloc[len(x_train):]))

    plot_acf(data['shock'])
    plt.show()
    plot_pacf(data['shock'])
    plt.show()
    sns.scatterplot(data['shock'], label='shock')
    sns.scatterplot(data['shock_pred'], label='pred')
    plt.show()
    sns.scatterplot(x=data['shock'], y=data['shock_pred'])
    plt.xlabel('Shock')
    plt.ylabel('Predictions')
    plt.show()