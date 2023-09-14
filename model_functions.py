import pandas as pd
import scipy.stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

########################### 

# FUNCTIONS FOR MODEL

###########################

def get_label(close, max_days):
    import time
    time_start = time.time()
    days = [max_days for i in range(len(close))]
    close_diff = close.diff().fillna(0)
    for i in range(len(close)):
        for j in range(i+1,i+max_days+1):
            condn = sum(close_diff[i+1:j]) >= 0.05 * close[i]
            if condn:
                days[i] = j-i-1
                break
    for i in range(len(days)):
        if days[i] == 0:
            days[i] += 20
    time_taken = time.time() - time_start
    print('label:',time_taken)
    return days

def get_reco(preds, max_days):
    have = False
    recos = []
    for pred in preds:
        if not have:
            if pred > max_days-1:
                recos.append(3) # ignore
            else:
                recos.append(1) # buy
                have = True
        else:
            if pred > max_days-1:
                recos.append(0) # sell
                have = False
            else:
                recos.append(2) # hold
    return recos   

def rss(actual, preds):
    n = len(actual)
    ans = 0
    for i in range(n):
        ans += (actual.iloc[i] - preds.iloc[i]) ** 2
    return ans

def tss(actual):
    n = len(actual)
    ans = 0
    for i in range(n):
        ans += (actual.iloc[i] - actual.mean()) ** 2
    return ans

def f_stat(actual, preds, p):
    rs, ts = 0 , 0
    n = len(preds)
    ts = tss(actual)
    rs = rss(actual, preds)
    f = ((ts - rs)/p) / (rs/(n-p-1))
    return f

def rse(actual, preds, p):
    rs = 0
    n = len(preds)
    rs = rss(actual, preds)
    rse = (rs/(n-p-1)) ** 0.5
    return rse

def r2(actual, preds):
    ts = tss(actual)
    rs = rss(actual, preds)
    r2 = 1 - rs/ts
    return r2    

def adj_r2(actual, preds, num_predictors):
    r2_ = r2(actual, preds)
    return 1 - (1-r2_) * (len(preds)-1) / (len(preds)-num_predictors-1)

def t_stat(coef, var, target, train_data, preds): 
    y = train_data[target]
    x = train_data[var]
    n = len(train_data)
    b = coef
    num, denom = 0,0
    for i in range(n):
        num += ((y.iloc[i] - preds.iloc[i])**2) / (n-2)
        denom += (x.iloc[i] - x.mean()) ** 2
    se = (num**0.5) / (denom**0.5)
    return b / se

def p_val(t, degf):
    return 2 * scipy.stats.t.sf(abs(t), df = degf)

def vif(features, train_data):
    vif_vals = {}
    model = LinearRegression()
    for i in range(len(features)):
        feat = features.copy()
        feat.pop(i)
        x = train_data[feat]
        y = train_data[features[i]]
        model.fit(x, y)
        preds = model.predict(x)
        preds = pd.Series(preds)
        r = r2(y, preds)
        vif = 1/(1-r)
        vif_vals[str(feat)] = vif
    return vif_vals

def corr(data, features):
    corr_list = {}
    for i in features:
        for j in features:
            corr_list[(i,j)] = data[i].corr(data[j])
    return corr_list

def get_return(data, reco):
    returns = [0 for i in range(len(data))]
    for i in range(len(reco)):
        close_initial, close_final = 0,0 
        if reco[i] == 1:
            close_initial = data['close'].iloc[i]
            for j in range(i+1, len(data)):
                if reco[j] == 0:
                    close_final = data['close'].iloc[j]
                    break
            returns[i] = close_final - close_initial
    return returns

def add_label_numperiods_qlearn(data, label, period_max):
    '''
    #add column for num_periods to reach t_return
    '''
    #import time
    #time_start = time.time()
    label_ = [period_max for i in range(len(data))]
    for i in range(len(data)):
        cum_sum = data.return_[i+1:i+period_max+1].cumsum()
        t_return = 0.05 * data['close'].iloc[i]
        inds = np.where(cum_sum >= t_return)[0]
        if len(inds) != 0:
            label_[i] = inds[0] + 1
    data[label] = label_
    #time_taken = time.time() - time_start
    #print(time_taken)    

def get_dataframes(model, features, dataframes, targets):
    for i in range(len(dataframes)):
        model.fit(dataframes[i][0][features], dataframes[i][0][targets[i]])
        preds_training = model.predict(dataframes[i][0][features])
        reco_training = get_reco(preds_training, 
                                max(dataframes[i][0][targets[i]]))
        preds_testing = model.predict(dataframes[i][1][features])
        reco_testing = get_reco(preds_testing,
                                max(dataframes[i][0][targets[i]]))
        preds_validation = model.predict(dataframes[i][2][features])
        reco_validation = get_reco(preds_validation, 
                                max(dataframes[i][0][targets[i]]))
        dataframes[i][0]['preds_mod_'+str(len(features))] = preds_training
        dataframes[i][0]['reco_mod_'+str(len(features))] = reco_training
        dataframes[i][0]['returns_mod_'+str(len(features))] = get_return(
            dataframes[i][0], reco_training)                                                                        
        dataframes[i][1]['preds_mod_'+str(len(features))] = preds_testing
        dataframes[i][1]['reco_mod_'+str(len(features))] = reco_testing
        dataframes[i][1]['returns_mod_'+str(len(features))] = get_return(
            dataframes[i][1], reco_testing)                                                                         
        dataframes[i][2]['preds_mod_'+str(len(features))] = preds_validation
        dataframes[i][2]['reco_mod_'+str(len(features))] = reco_validation
        dataframes[i][2]['returns_mod_'+str(len(features))] = get_return(
            dataframes[i][2], reco_validation)
     
    return dataframes

def odds_profit(data, reco, target_col, max_days):
    odds_list = []
    count_success=0
    count_failure=0
    days_returns = get_return(data, reco)
    days_list = [i for i in range(max_days)]
    for i in days_list:
        for j in range(len(data)):
            if data[target_col].iloc[j]==i:
                if days_returns[j]>0:
                    count_success+=1
                else:
                    count_failure+=1
        if count_success==0 and count_failure==0:
            odds_list.append(0)
        else:
            odds_success = count_success/(count_success+count_failure)
            odds_list.append(odds_success/(1-odds_success))
    return odds_list

def segement_by_date(data, occupancy_vals, date):
    data_date = data[data['Date'] == date]
    data_date_0 = data_date[data_date['Room_Occupancy_Count']==
                            occupancy_vals[0]]
    data_date_1 = data_date[data_date['Room_Occupancy_Count']==
                            occupancy_vals[1]]
    data_date_2 = data_date[data_date['Room_Occupancy_Count']==
                            occupancy_vals[2]]
    data_date_3 = data_date[data_date['Room_Occupancy_Count']==
                            occupancy_vals[3]]
    return data_date_0, data_date_1, data_date_2, data_date_3

def plot_vals_vs_time(data, plot_cols, dates, plot_type = sns.lineplot):
    for _date in dates:
        for _item in plot_cols:
            plot_type(x = data['Hour'][data['Date']==_date],
                         y = data[_item][data['Date']==_date],label=_item)
            
        plt.title(label=_date)
        plt.ylabel('Measured Values')
        plt.show()
        
def plot_people_vs_vals(data, plot_cols, dates):
    for _date in dates:
        for _col in plot_cols:
            sns.scatterplot(
                y = data['Room_Occupancy_Count'][data['Date']==_date], 
                x = data[_col][data['Date']==_date],
                label=_col
                )
            plt.title(label=_date)
            plt.xlabel('Measured Values')
            plt.ylabel('People')
            plt.show()

def round_items(items):
    new = []
    for item in items:
        if item - int(item) < 0.5:
            new.append(int(item))
        else:
            new.append(int(item)+1)
    return new

def accuracy(actual, preds):
    count = 0
    for i in range(len(actual)):
        if actual.iloc[i]==preds.iloc[i]:
            count+=1
    return count/len(actual)

def zero_one_split(data, col):
    data_zero = data[data[col] == 2]
    data_one = data[data[col] == 3]
    return data_zero, data_one

def draw_plots(data, target, cols):
    for col in cols:
        sns.scatterplot(data, x = col, y = target)
        plt.show()
    for col in cols:
        plt.title(col)
        sns.boxplot(data[col])
        plt.show()
    for col in cols:
        sns.histplot(data[col])
        plt.show()
    return None

def handle_missing_vals(data, cols, method = 'drop'):
    # method can have values 'drop' and 'replace'
    if method == 'drop':
        for col in cols:
            data = data[data[col]!=np.nan]
        return None
    if method == 'replace':
        choice = int(input('1. Mean, 2. Median, 3. Mode, 4. Custom: '))
        for col in cols:
            if choice == 1:
                data.loc[data[col] == np.nan] = data[col].mean()
            elif choice == 2:
                data.loc[data[col] == np.nan] = data[col].median()
            elif choice == 3:
                data.loc[data[col] == np.nan] = data[col].mode()[0]
            elif choice == 4:
                fill_val = int(input('Enter value to fill with: '))
                data.loc[data[col] == np.nan] = fill_val
    return None

def handle_outliers(data, cols, method = 'drop'):
    # method can have values drop and replace
    for col in cols:
        q1 = data[col].quantile(q=.25)
        q3 = data[col].quantile(q=.75)
        iqr = q3 - q1
        lower_lim = q1 - 1.5* iqr
        upper_lim = q3 + 1.5* iqr
        if method == 'drop':
            data = data[data[col].between(lower_lim, upper_lim)].reset_index(
                drop=True)
        if method == 'replace':
            data.loc[data[col] > upper_lim, col] = upper_lim
            data.loc[data[col] < lower_lim, col] = lower_lim
    return data

def standardize(data, cols):
    for col in cols:
        method = int(input('1. Min-Max, 2. Decimal Scaling, 3. Z Score: '))
        if method == 1:
            print(col)
            newmin = int(input('Enter new_min: '))
            newmax = int(input('Enter new_max: '))
            denom = max(data[col]) - min(data[col])
            data[col] = (((data[col]-min(data[col]))/denom)*(newmax-newmin))
            +newmin
        elif method == 2:
            e = int(input('Enter power of 10 to divide by: '))
            data[col] = data[col]/10**e
        elif method == 3:
            data[col] = (data[col] - data[col].mean()) / data[col].std()
    return None

def group_continuous_data(data, col, bins, names):
    new_name = str(col) + '_cat'
    data[new_name] = pd.cut(data[col], bins, labels = names)
    return None

def fisher_score(data, cols):
    scores = []
    zeros, ones = zero_one_split(data, 'Room_Occupancy_Count') 
    for col in cols:
        zeros_mean = zeros[col].mean()
        zeros_var = zeros[col].std()**2
        ones_mean = ones[col].mean()
        ones_var = ones[col].std()**2
        scores.append(abs((ones_mean - zeros_mean)) / ((ones_var - zeros_var)
                                                  ** 0.5))
    dataframe = pd.DataFrame({'columns':cols, 'fisher_score':scores})
    return dataframe

def correlation_scores(data, target, features):
    scores = []
    for feature in features:
        scores.append(data[feature].corr(data[target]))       
    dataframe = pd.DataFrame({'columns':features, 'corr_score':scores})
    return dataframe
