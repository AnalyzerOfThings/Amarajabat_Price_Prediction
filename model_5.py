import pandas as pd
import statsmodels.api as sm
import functions
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from ta.trend import macd_diff, macd

################ LOADING FILE

path = 'amarajabat.xlsx'
data = pd.read_excel(path)

################ CALCULATING MACD AND ITS FIRST ORDER DIFFERENCE

macd_diff = macd_diff(data['close'], fillna= True)
macd = macd(data['close'], fillna = True)
close = data['close']
close_diff = close.diff().fillna(0)

################ CALCULATE VALUE OF LABEL (DAYS)

days = functions.get_label(close)


################ ADDING FEATURES TO DATAFRAME

data['macd'] = macd
data['macd_diff'] = macd_diff
data['change_lag_1'] = ((close.shift(1)/close)-1)
data['change_lag_2'] = ((close.shift(2)/close)-1)
data['change_lag_5'] = ((close.shift(5)/close)-1)
data['days'] = days

data['change_lag_1'] = data['change_lag_1'].fillna(data['change_lag_1'].mean())
data['change_lag_2'] = data['change_lag_2'].fillna(data['change_lag_2'].mean())
data['change_lag_5'] = data['change_lag_5'].fillna(data['change_lag_5'].mean())

############### CLEANING DATA

data = functions.drop_nan(data, 'days', 0)
data = functions.log_target(data, 'days')
data = functions.drop_outliers(data, 'log_days')

############## MODEL

features = ['macd','macd_diff', 'change_lag_1', 'change_lag_2', 'change_lag_5']
train_test_data = data[:1600]
validate_data = data[1600:]

x = train_test_data[features]
y = train_test_data.log_days
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=1000,
                                                    shuffle=False)

model_sk = LinearRegression()
model_sk.fit(x_train, y_train)
preds_testing = model_sk.predict(x_test)
mae = mean_absolute_error(y_test, preds_testing)
print('MAE: ',mae)

x_train_ = sm.add_constant(x_train)
model_sm = sm.OLS(y_train,x_train_)
res = model_sm.fit()
print('REGRESSED ON TRAINING DATA')
print(res.summary())

print('r squared score of testing data: ',r2_score(y_test, preds_testing))

############# VALIDATION DATA

x_validation = validate_data[features]
y_validation = validate_data.log_days
preds_validation = model_sk.predict(x_validation)

print('r squared score of validation data: ',r2_score(y_validation,
                                                      preds_validation))