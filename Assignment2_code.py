#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 13:13:45 2023

@author: M
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import boxcox
from scipy.special import inv_boxcox

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import pmdarima as pm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing   
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from plot_helper import plot, plot_decompose

# read data and format
df = pd.read_csv('drug.txt')
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df = df.set_index(df['date'])
df = df.drop(columns=['date'])
print(df.head())        # visualize first 5 rows
print(df.tail())        # visualize last 5 rows
print(len(df.index))    # Get number of entries
print(df.dtypes)        # See datatype of entries
print(df[df.isnull().any(axis=1)])

# original time series plot
plot(df["value"],
     title=r'Anti-diabetic Drug Sales',
     x_label=r'Year',
     y_label=r'Drug Sales')

# seasonal decomposition
result = seasonal_decompose(df, model='additive')
plot_decompose(result)

# acf and pacf of original data
plot_acf(df)
plot_pacf(df)

# boxcox transform
df['box_cox_value'], lambda_val = boxcox(df['value'])
print("Lambda value:", lambda_val)

# plot after boxcox transform
plot(df["box_cox_value"],
     title=r'BoxCox Transform',
     x_label=r'Year',
     y_label=r'Drug Sales')

# one time differencing of data (trend)
df_one_diff = df.diff()

# plot one time differenced data
plot(df_one_diff["box_cox_value"],
     title=r'One Time Differencing for Trend',
     x_label=r'Year',
     y_label=r'Drug Sales')

# one time differencing of data (seasonal)
df_one_diff = df.diff(periods=12)

# plot one time differenced data
plot(df_one_diff["box_cox_value"],
     title=r'Differencing for Season',
     x_label=r'Year',
     y_label=r'Drug Sales')

# drop value column for plotting acf and pacf
df_one_diff = df_one_diff.drop(columns=['value'])
# drop nan rows after differencing
df_one_diff = df_one_diff.iloc[12:]

# acf and pacf of transformed data
plot_acf(df_one_diff, lags=65)
plot_pacf(df_one_diff, lags=65)

# train test split
train, test = train_test_split(df, test_size=0.2, shuffle=False)

# =============================================================================
# SARIMA model
# SARIMA(p,d,q)(P,D,Q)m
# =============================================================================
sarima = pm.auto_arima(train["box_cox_value"], start_p=1, start_q=1,
                         test='adf',
                         max_p=5, max_q=4, 
                         m=12, # 12 is the frequncy of the cycle
                         start_P=0, 
                         seasonal=True, # set to seasonal arima
                         d=1, # apply one time differencing for seasonality
                         D=1,
                         trace=False,
                         error_action='warn',  
                         suppress_warnings=True, 
                         stepwise=True)

print(sarima.summary())
sarima.plot_diagnostics(figsize=(15,12))
plt.show()

def forecast(sarima, periods=80):
    # Forecast
    fitted, confint = sarima.predict(n_periods=80, return_conf_int=True)
    index_of_fc = pd.date_range(train.index[-1] + pd.DateOffset(months=1), periods = periods, freq='MS')

    # add values to series and invert boxcox
    fitted_series = inv_boxcox(pd.Series(fitted, index=index_of_fc), lambda_val)
    lower_series = inv_boxcox(pd.Series(confint[:, 0], index=index_of_fc), lambda_val)
    upper_series = inv_boxcox(pd.Series(confint[:, 1], index=index_of_fc), lambda_val)

    # Plot
    plt.figure(figsize=(15,7))
    train["value"].plot(legend=True, label='train')
    test["value"].plot(legend=True, label='test')
    fitted_series.plot(legend=True, label='prediction')
    plt.fill_between(lower_series.index, 
                    lower_series, 
                    upper_series, 
                    color='k', alpha=.15)

    plt.title("SARIMA forecast")
    plt.show()
    
forecast(sarima)

# RMSE calculation
rmse_preds, confint = sarima.predict(n_periods=len(test.index), return_conf_int=True)
rmse_preds = inv_boxcox(rmse_preds, lambda_val)
print("Test RMSE: %.3f" % np.sqrt(mean_squared_error(test['value'], rmse_preds)))

# =============================================================================
# Holt Winters' Model
# =============================================================================

# define x as time period and set alpha
x = 12
alpha = 1/(2*x)

# single exponential smoothing
df['HWES1'] = SimpleExpSmoothing(df['box_cox_value']).fit(smoothing_level=alpha,optimized=False,use_brute=True).fittedvalues      
df[['box_cox_value','HWES1']].plot(title='Holt Winters Single Exponential Smoothing Graph')
plt.show()

# double exponential smoothing - additive and multiplicative
df['HWES2_ADD'] = ExponentialSmoothing(df['box_cox_value'],trend='add').fit().fittedvalues
df['HWES2_MUL'] = ExponentialSmoothing(df['box_cox_value'],trend='mul').fit().fittedvalues
df[['box_cox_value','HWES2_ADD','HWES2_MUL']].plot(title='Holt Winters: Additive Trend and Multiplicative Trend')
plt.show()

# fit the model
hw_model = ExponentialSmoothing(train["box_cox_value"],trend='mul',seasonal='mul',seasonal_periods=x).fit()
hw_model_damped = ExponentialSmoothing(train["box_cox_value"],trend='mul',seasonal='mul',seasonal_periods=x, damped=True).fit()
test_predictions = inv_boxcox(hw_model.forecast(80), lambda_val)
test_predictions_damped = inv_boxcox(hw_model_damped.forecast(80), lambda_val)
train['value'].plot(legend=True,label='train')
test['value'].plot(legend=True,label='test',figsize=(15,7))
test_predictions.plot(legend=True,label='prediction')
test_predictions_damped.plot(legend=True,label='damped_prediction')
plt.title('Holt Winters Exponential Smoothing - Train, Test, Predicted')
plt.show()

# model summary
print(hw_model.summary())
print(hw_model_damped.summary())

# RMSE calculation
rmse_test_predictions = inv_boxcox(hw_model.forecast(len(test.index)), lambda_val)
rmse_test_predictions_damped = inv_boxcox(hw_model_damped.forecast(len(test.index)), lambda_val)

print("Test RMSE: %.3f" % np.sqrt(mean_squared_error(test['value'], rmse_test_predictions)))
print("Test RMSE - damped: %.3f" % np.sqrt(mean_squared_error(test['value'], rmse_test_predictions_damped)))




