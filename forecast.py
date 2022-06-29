#import requests
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from datetime import datetime

from pandas import Series

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import SimpleExpSmoothing, VAR
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from pmdarima import auto_arima

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.preprocessing import MinMaxScaler

torch.manual_seed(1)

data_dir = '.'
TRAIN_PERC = .8

def plot_data(purchased, sold):
    #################################################
    ###  Histogram of purchased and sold resources
    #################################################
    tot_purchased = purchased.groupby('resource_name').purchased.sum()
    tot_sold = sold.groupby('resource_name').sold.sum()
    totals = pd.concat([tot_sold,tot_purchased], axis=1)
    totals.loc[pd.isna(totals.purchased),'purchased'] = 0
    totals.loc[pd.isna(totals.sold),'sold'] = 0

    # Plot settings
    X_axis = np.arange(len(totals))
    width = 0.25
    # create a figure to show plots on different windows in a non-blocking manner
    solpur_fig = plt.figure(num='solpur')
    ax1 = solpur_fig.add_subplot(1,1,1)
    ax1.bar(X_axis-width/2, totals.purchased, width=width, color='r', label='purchased')
    ax1.bar(X_axis+width/2, totals.sold, width=width, color='g', label='sold')
    ax1.legend(loc = 'best')
    
    plt.xticks(X_axis, totals.index, rotation='vertical')
    plt.xlabel("Resource")
    plt.ylabel("Kg")
    plt.title("Resources sold and purchased")
    # Show but don't block
    solpur_fig.show()

    #####################################################
    ###  Histogram of purchased and sold datapoint counts
    #####################################################

    count_purchased = purchased.groupby('resource_name').purchased.count()
    count_sold = sold.groupby('resource_name').sold.count()
    totals = pd.concat([count_sold,count_purchased], axis=1)
    totals.loc[pd.isna(totals.purchased),'purchased'] = 0
    totals.loc[pd.isna(totals.sold),'sold'] = 0

    # Plot settings
    X_axis = np.arange(len(totals))
    width = 0.25
    # create a figure to show plots on different windows in a non-blocking manner
    counts_fig = plt.figure(num='counts_fig')
    ax1 = counts_fig.add_subplot(1,1,1)
    ax1.bar(X_axis-width/2, totals.purchased, width=width, color='r', label='purchased')
    ax1.bar(X_axis+width/2, totals.sold, width=width, color='g', label='sold')
    ax1.legend(loc = 'best')
    
    plt.xticks(X_axis, totals.index, rotation='vertical')
    plt.xlabel("Resource")
    plt.ylabel("Datapoints")
    plt.title("Datapoint counts on resources sold and purchased")
    # Show but don't block
    counts_fig.show()
    
    plt.show()
    plt.close(fig=solpur_fig)
    plt.close(fig=counts_fig)

def create_in_out_seq(input_data, tw):
    inout_seq = []
    l = len(input_data)
    for i in range(l-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def reset_hidden(self):
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

def train_lstm(train_inout_seq, model, optimizer,loss_function, epochs=200):

    for i in range(epochs):
        for seq, labels in train_inout_seq:
            labels = labels.squeeze()
            optimizer.zero_grad()
            model.reset_hidden()

            y_pred = model(seq)
            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

        if i%25 == 1:
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

    print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

def make_pred(model, fut_pred, train_window, test_inputs):
    model.eval()

    for i in range(fut_pred):
        seq = torch.FloatTensor(test_inputs[-train_window:])
        with torch.no_grad():
            model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size))
            test_inputs.append(model(seq).tolist())
            # breakpoint()
    
    return test_inputs[-fut_pred:]

def df_test(resource, time_series):
    # Advanced Dickey-Fuller Test for stationariety
    # print(f'Results of Dickey-Fuller Test for {resource}:')
    dftest = adfuller(time_series)
    
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    
    if dfoutput['p-value'] < .05 :
        # print(f"{resource} data is stationary")    
        return True
    
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' %key] = value
    
    print(f"{resource} data is NOT stationary")
    # print (dfoutput)
    return False

def ses(train_set):
    ## auto optimization
    model = SimpleExpSmoothing(train_set)
    fit = model.fit()
    # print(fit.summary())
    return fit

def arima(train_set):
    model = ARIMA(train_set, order=(2, 1, 0))
    fit = model.fit()
    # print(fit.summary())
    
    return fit

def sarima(train_set, m):
    model = auto_arima(train_set,
                        start_p=0,
                        start_q=0,
                        max_p=3,
                        max_q=3,
                        d=None,
                        D=None,
                        max_d=3,
                        start_P=0,
                        start_Q=0,
                        max_P=3,
                        max_D=3,
                        max_Q=3,
                        seasonal=True,
                        m=m,
                        stepwise=True,
                        trace=False)

    fit = model.fit(train_set)
    # print(fit.summary())
    return fit.arima_res_

def autoreg(train_set, test_set):
    """
        Loop on all resources and calculate
        - Simple Exponential Smoothing
        - ARIMA
        - SARIMA
    """
    #################################################
    ###  Loop on all sold resources
    #################################################
    
    best_mse = 100000
    best_mse_method = None
    best_resource = None
    best_ses_fcast = None
    best_arima_fcast = None
    best_sarima_fcast = None

    for resource in train_set.columns:
        # select and resample per day
        train_resource = train_set.loc[:,resource]
        test_resource = test_set.loc[:,resource]
     
        updated = False
        ses_fcast = None
        arima_fcast = None
        sarima_fcast = None
        # Test if data is stationary
        if df_test(resource, train_resource):
            
            ses_fit = ses(train_resource)
            ses_fcast = ses_fit.forecast(len(test_resource))
            mse2 = ((ses_fcast - test_resource) ** 2).mean()
            # avoid trivial case where future sales are all zero
            if max(test_resource) > 0 and best_mse > mse2:
                best_mse = mse2
                best_resource = resource
                best_mse_method = 'ses'
                updated = True
            # print(f"SES RMS Error (smoothing = {ses_fit.model.params['smoothing_level']}) is {round(np.sqrt(mse2), 2)}")
            
            arima_fit = arima(train_resource)
            arima_fcast = arima_fit.forecast(len(test_resource))
            mse2 = ((arima_fcast - test_resource) ** 2).mean()
            # avoid trivial case where future sales are all zero
            if max(test_resource) > 0 and best_mse > mse2:
                best_mse = mse2
                best_resource = resource
                best_mse_method = 'arima'
                updated = True
            # print(f"ARIMA RMS Error is {round(np.sqrt(mse2), 2)}")
            
        sarima_fit = sarima(train_resource,7)
        sarima_fcast = sarima_fit.forecast(len(test_resource))
        mse2 = ((sarima_fcast - test_resource) ** 2).mean()
        # avoid trivial case where future sales are all zero
        if max(test_resource) > 0 and best_mse > mse2:
                best_mse = mse2
                best_resource = resource
                best_mse_method = 'sarima'
                updated = True
        # print(f"SARIMA RMS Error is {round(np.sqrt(mse2), 2)}")
        if updated:
            best_ses_fcast = ses_fcast
            best_arima_fcast = arima_fcast
            best_sarima_fcast = sarima_fcast
        
    print(f'Best MSE is {best_mse} with {best_mse_method}')
    solres_fig = plt.figure(num=best_resource)
    ax1 = solres_fig.add_subplot(1,1,1)

    ax1.plot(test_set.loc[:,best_resource], ".-", color='g', label = best_resource)
    
    plt.xlabel("Time(day)", fontsize=10)
    plt.ylabel("Kg", fontsize=16)
    plt.title(f'Daily {best_resource} sold', fontsize=16)
    solres_fig.show()
    
    best_ses_fcast = pd.DataFrame(best_ses_fcast,columns=[best_resource],index=test_set.index)
    ax1.plot(best_ses_fcast, ".-", color='r', label = 'ses')
    
    best_arima_fcast = pd.DataFrame(best_arima_fcast,columns=[best_resource],index=test_set.index)
    ax1.plot(best_arima_fcast, ".-", color='b', label = 'arima')

    best_sarima_fcast = pd.DataFrame(best_sarima_fcast,columns=[best_resource],index=test_set.index)
    ax1.plot(best_sarima_fcast, ".-", color='y', label = 'sarima')
    
    # Plot autocorrellation and partial autocorrelation
    acf_plot = plot_acf(train_set.loc[:,best_resource])
    plt.title(f'{best_resource} autocorrelation', fontsize=16)
    acf_plot.show()
    pacf_plot = plot_pacf(train_set.loc[:,best_resource], method='ywm')
    plt.title(f'{best_resource} partial autocorrelation', fontsize=16)
    pacf_plot.show()

    # Plot seasonal decomposition
    seasonal_plot = sm.tsa.seasonal_decompose(train_set.loc[:,best_resource]).plot()
    seasonal_plot.show()
    
    
    ax1.legend(loc = 'best')
    plt.show()
    plt.close(fig=solres_fig)
    plt.close(fig=acf_plot)
    plt.close(fig=pacf_plot)
    plt.close(fig=seasonal_plot)

    # breakpoint()

def vect_ar(sold):
    """
        Fit a Vector AR
    """
    model = VAR(sold)
    model_fit = model.fit(maxlags=5)
    model_fit.summary()

def dateparse(d,t):
    dt = d + " " + t
    return pd.to_datetime(dt, format='%Y-%m-%d %I:%M %p')

def read_data():
    purchased = pd.read_csv(f'{data_dir}/purchased.csv', 

                            sep = ',',

                            dtype={'resource_id':'string',
                                    'resource_name':'string',
                                    'unit':'string',
                                    'purchased':'float64'
                                    },
                                
                            parse_dates={'datetime': ['date', 'time']}, 
                                
                            date_parser=dateparse)

    sold = pd.read_csv(f'{data_dir}/sold.csv', 

                            sep = ',',

                            dtype={'resource_id':'string',
                                    'resource_name':'string',
                                    'unit':'string',
                                    'sold':'float64'
                                    },
                                
                            parse_dates={'datetime': ['date', 'time']}, 
                                
                            date_parser=dateparse)

    shelf_life = pd.read_csv(f'{data_dir}/shelf_life.csv', 

                            sep = ';',

                            dtype={'resource_name':'string',
                                    'shelf_life_days':'int16'
                                    }
                            )
    return purchased, sold, shelf_life

def main():
    purchased, sold, shelf_life = read_data()
    sold.index = sold.datetime
    sold.drop(['unit','resource_id','datetime'],axis=1, inplace=True)
    purchased.index = purchased.datetime
    purchased.drop(['unit','resource_id','datetime'],axis=1, inplace=True)

    # plot_data(purchased, sold)

    sold_allres =  sold.set_index('resource_name',append=True).unstack().resample('D').sum()
    sold_allres.columns = sold_allres.columns.get_level_values(1)


    ### Splitting train and test data
    data_len = len(sold_allres)
    train_set = sold_allres.iloc[:int(data_len*TRAIN_PERC),:]
    test_set = sold_allres.iloc[int(data_len*TRAIN_PERC):,:]
    print(f'predicting {len(test_set)} days')
    
    # breakpoint()
    # vect_ar(sold_allres)
    autoreg(train_set, test_set)

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_set_scaled = scaler.fit_transform(train_set)
    train_set_scaled = torch.FloatTensor(train_set_scaled)
    
    train_window = 7
    train_inout_seq = create_in_out_seq(train_set_scaled, train_window)

    feature_size = sold_allres.shape[1]

    model = LSTM(input_size=feature_size,output_size=feature_size )
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_lstm(train_inout_seq, model, optimizer,loss_function)
    
    predictions = make_pred(model, fut_pred=len(test_set), train_window=train_window, test_inputs=train_set_scaled[-train_window:].tolist())
    
    actual_predictions = scaler.inverse_transform(predictions)

    actual_predictions = pd.DataFrame(actual_predictions,columns=test_set.columns,index=test_set.index)

    ### Calculate errors per product and plot the best product
    mse = (np.square(test_set - actual_predictions)).mean(axis=0)
    print('MSE per resource')
    print(mse)
    best_index = list(mse).index(min(mse))
    respred_fig = plt.figure(num='predictions')
    ax1 = respred_fig.add_subplot(1,1,1)
    ax1.plot(test_set.iloc[:,best_index], ".-", color='g', label = test_set.iloc[:,best_index].name)
    ax1.plot(actual_predictions.iloc[:,best_index], ".-", color='r', label = 'product pred')
    ax1.legend(loc = 'best')

    plt.xlabel("Time(day)", fontsize=10)
    plt.ylabel("Kg", fontsize=16)
    plt.title(f'Daily {test_set.iloc[:,best_index].name} sold', fontsize=16)
    respred_fig.show()
   
    ### Calculate errors per day and plot the best day
    mse = (np.square(test_set - actual_predictions)).mean(axis=1)
    print('MSE per day')
    print(mse)
    best_index = list(mse).index(min(mse))
    daypred_fig = plt.figure(num='daypred')
    ax1 = daypred_fig.add_subplot(1,1,1)

    ax1.plot(test_set.iloc[best_index,:], ".-", color='g', label = test_set.iloc[best_index,:].name)
    ax1.plot(actual_predictions.iloc[best_index,:], ".-", color='r', label = 'day pred')
    ax1.legend(loc = 'best')

    plt.xticks(np.arange(len(test_set.columns)), test_set.columns, rotation='vertical')
    plt.xlabel("Resource", fontsize=10)
    plt.ylabel("Kg", fontsize=16)
    plt.title(f'Sales on {test_set.iloc[best_index,:].name}', fontsize=16)
    plt.show()

    plt.close(fig=respred_fig)
    plt.close(fig=daypred_fig)
    

      

    
    
    
    



if __name__ == "__main__":
    import argparse
    from six import text_type
    import sys
    

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # parser.add_argument(
    #     'suffix',
    #     type=text_type,
    #     nargs=1,
    #     help='specifies the suffix of the file to consider',
    # )

    args, unknown = parser.parse_known_args()

    try:
        main(
            # args.suffix[0],
        )
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("Done\n")


