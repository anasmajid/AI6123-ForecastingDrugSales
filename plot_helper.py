#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 13:50:10 2023

@author: M
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot(y, title=r'title', x_label=r'xlabel', y_label=r'ylabel'):
    
    mpl.rcParams['figure.dpi'] = 600
    fig, ax = plt.subplots(1, 1)
    
    ax.set_facecolor('.85')
    ax.plot(y, linewidth=0.5, color='blue')
    
    # set monthly locator
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=24))
    # set formatter
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # set font and rotation for date tick labels
    plt.gcf().autofmt_xdate()
    
    ax.grid(color='white', linewidth=0.5)
    
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    plt.show()
    
def plot_decompose(decom_result):
    fig, axes = plt.subplots(4, 1, sharex=True)
    
    decom_result.observed.plot(ax=axes[0], legend=False, color='k')
    axes[0].set_ylabel('Observed')
    decom_result.trend.plot(ax=axes[1], legend=False, color='r')
    axes[1].set_ylabel('Trend')
    decom_result.seasonal.plot(ax=axes[2], legend=False, color='g')
    axes[2].set_ylabel('Seasonal')
    decom_result.resid.plot(ax=axes[3])
    axes[3].set_ylabel('Residual')
    
    for ax in axes:
        ax.grid(color='white', linewidth=0.5)
    
    plt.show()
    

    