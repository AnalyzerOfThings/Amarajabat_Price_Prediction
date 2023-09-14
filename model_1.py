import pandas as pd
import statsmodels.api as sm
from ta.trend import macd_diff, macd
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_absolute_error as mae
from sklearn.linear_model import LinearRegression



################ LOADING FILE

path = 'amarajabat.xlsx'
data = pd.read_excel(path)

################ CALCULATING MACD AND ITS FIRST ORDER DIFFERENCE

macd_diff = macd_diff(data['close'], fillna= True)
macd = macd(data['close'], fillna = True)


####################################################

'''
DECIDING WHAT DO DO ON BASIS OF THE RULES:
    
    if have the stock:
        if +5% within 20 days, hold
        else sell
    if not have stock:
        if +5% within 20 days, buy
        else ignore
    
'''
####################################################

have = False
length = len(macd_diff)
decision = []
for i in range(length):
    condn = sum(macd_diff.iloc[i:i+20]) >= 0.05* macd[i]
    if not have:
        if condn:
            decision.append('buy')
            have = True
        else:
            decision.append('ignore')
    else:
        if condn:
            decision.append('hold')
        else:
            decision.append('sell')
            have = False

####################################################
            
'''
COUNTING NUMBER OF DAYS TO HOLD BASED ON THE DECISIONS MADE ABOVE:
    
    if decision == sell or ignore:
        number of days to hold = 0
    
    if decision == buy or hold:
        number of days = index_of_closest_sell_decision - current_index
'''

####################################################

days=[]
sell_ind = 0
count=0
for i in range(len(decision)):
    if decision[i] == 'sell' or decision[i] == 'ignore':
        days.append(0)
    else:
        for j in range(i, len(decision)):
            if decision[j] == 'sell':
                count+=1
                sell_ind = j
                days.append(sell_ind-i)
                break

days = pd.Series(days)

####################################################


# CREATING A NEW DATAFRAME WITH THE CALCULATED VALUES FOR SIMPLICITY

new_data = pd.DataFrame({'macd':macd, 'macd_diff':macd_diff,'days':days})

####################################################

# BUILDING MODEL [USING STATSMODELS.API BECAUSE THE REGRESSION SUMMARY IS CLEAR AND COMPLETE]

####################################################

data = new_data.astype(float)
features = ['macd','macd_diff']
limit = int(0.75 * len(data))

x = data[features]
y = data.days

model = sm.OLS(y,x)
res = model.fit()
print(res.summary())

#####################################################

# USING SKLEARN

#####################################################

model = LinearRegression()
train_x, test_x, train_y, test_y = tts(x,y,shuffle=False)
model.fit(train_x, train_y)
preds = model.predict(test_x)
mae = mae(test_y, preds) # 9.458