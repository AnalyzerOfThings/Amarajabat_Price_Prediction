# Amarajabat_Price_Prediction
Predicting share prices for Amarajabat using open, low, high, close, and volume data.


I have tried using several models to find a solution to the problem. I started this project
while I was beginning to learn Data Science, and as a result, I've made several models, each one
showcasing what I learned in the time between. I've uploaded all of these, mainly to show what
I learned over a two month period. 

A description of each of the uploaded files follows :

1. amarajabat.xlsx : an excel file containing the dataset. you may have to change it into a csv file for some of the
   files.
2. model_2.py : Linear regression is used to find the number of days taken for the share price to increase by 5%.
   if this number is below 20, the model purchases the stock.
3. model_3.py : Linear regression, but additional features, namely, the return over the last 1, 2, and 5 days respectively,
   have been added.
4. model_4.py : The same as above, with other improvements.
5. model_5.py : More improvements to the methodolgy.
6. model_6.py : Instead of deciding to buy if the target < 20, different models have been made for the target < 50 and
   target < 100. The performance of these models has been compared.
7. model_functions.py : Contains several utility functions. Some of which are unrelated to this project.
8. stock_time_series.py : Return is predicted. As it is stationary, I've used AR, MA, ARMA, ARIMA, as well as linear regression
   to predict the target.
9. time_series_manual.py : The pacf plot of return (over five days) shows an interesting pattern. I made an ARMA model to showcase
   this. The ARMA model is made using the LinearRegression() class.
10. time_series_manual_2 : Unfortunately, in the above model, the features [shock_shifted_1, shock_shifted_2, shock_shifted_3, shock_shifted_4]
    are unknown at the day of making the prediction. But that's fine, because shock_shifted_5 and shock_shifted_6 show decent pacf values with the
    target. Unfortunately, the performance drops severely. 
11. time_series_edited.py : This model cleans up the last one, and a few core conceptual mistakes have been rectified. Other ML techniques have also been
    used here.

Basic preprocessing and visualization has been done in each of the individual files themselves. 
It should be noted that the poor results of the models are expected. Stock data is highly random, as can be seen by the plots of return.
Thus, making good predictions is very tough. I chose this problem because I knew a simple problem wouldn't motivate me to try new ideas out
to make a better model.
