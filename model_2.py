import pandas as pd
import statsmodels.api as sm
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

days = [36500 for i in range(len(close))]
for i in range(len(close)):
    for j in range(i+1,len(close)):
        condn = sum(close_diff[i+1:j]) >= 0.05 * close[i]
        if condn:
            days[i] = j-i-1
            break

# CREATING A NEW DATAFRAME WITH THE CALCULATED VALUES FOR SIMPLICITY

new_data = pd.DataFrame({'macd':macd, 'macd_diff':macd_diff,'days':days})

####################################################

# BUILDING MODEL

####################################################

data = new_data.astype(float)
features = ['macd','macd_diff']
limit = int(0.75 * len(data))

x = data[features]
y = data.days

model = sm.OLS(y,x)
res = model.fit()
print(res.summary())        