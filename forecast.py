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

data_dir = '.'
TRAIN_PERC = .8

def df_test(resource, time_series):
    # Advanced Dickey-Fuller Test for stationariety
    print(f'Results of Dickey-Fuller Test for {resource}:')
    dftest = adfuller(time_series)
    
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    
    if dfoutput['p-value'] < .05 :
        print(f"{resource} data is stationary")    
        return True
    
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' %key] = value
    
    print(f"{resource} data is NOT stationary")
    print (dfoutput)
    return False

def ses(y_to_train):
    ## auto optimization
    model = SimpleExpSmoothing(y_to_train)
    fit = model.fit()
    print(fit.summary())
    return fit

def arima(y_to_train):
    model = ARIMA(y_to_train, order=(2, 1, 0))
    fit = model.fit()
    print(fit.summary())
    
    return fit

def sarima(y_to_train, m):
    model = auto_arima(y_to_train,
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

    fit = model.fit(y_to_train)
    print(fit.summary())
    return fit.arima_res_

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
    
    plt.xticks(X_axis, totals.index, rotation='vertical')
    plt.xlabel("Resource")
    plt.ylabel("Kg")
    plt.title("Resources sold and purchased")
    # Show but don't block
    solpur_fig.show()
    
    #################################################
    ## Fit a Vector AR
    #################################################
    sold_allres =  sold.set_index('resource_name',append=True).unstack().resample('D').sum()
    model = VAR(sold_allres)
    model_fit = model.fit(maxlags=5)
    model_fit.summary()
    
    #################################################
    ###  Loop on all sold resources
    #################################################
    for resource in sold['resource_name'].unique():
        # select and resample per day
        sold_resource = sold.loc[sold['resource_name']==resource].resample('D').sum()

        solres_fig = plt.figure(num=resource)
        ax1 = solres_fig.add_subplot(1,1,1)
        ax1.plot(sold_resource['sold'], ".-", color='g', label = resource)
        
        plt.xlabel("Time(day)", fontsize=10)
        plt.ylabel("Kg", fontsize=16)
        plt.title(f'Daily {resource} sold', fontsize=16)
        solres_fig.show()
   
        # Plot autocorrellation and partial autocorrelation
        acf_plot = plot_acf(sold_resource.sold)
        plt.title(f'{resource} autocorrelation', fontsize=16)
        acf_plot.show()
        pacf_plot = plot_pacf(sold_resource.sold, method='ywm')
        plt.title(f'{resource} partial autocorrelation', fontsize=16)
        pacf_plot.show()

        # Plot seasonal decomposition
        seasonal_plot = sm.tsa.seasonal_decompose(sold_resource.sold).plot()
        seasonal_plot.show()
        
        # Splitting train and test data
        data_len = len(sold_resource)
        y_to_train = sold_resource[:int(data_len*TRAIN_PERC)].squeeze(axis=1)
        y_to_test = sold_resource[int(data_len*TRAIN_PERC):].squeeze(axis=1)
        print(f'predicting {len(y_to_test)} days')
        
        # Test if data is stationary
        if df_test(resource, sold_resource):
            
            ses_fit = ses(y_to_train)
            ses_fcast = ses_fit.forecast(len(y_to_test))
            mse2 = ((ses_fcast - y_to_test) ** 2).mean()
            print(f"SES RMS Error (smoothing = {ses_fit.model.params['smoothing_level']}) is {round(np.sqrt(mse2), 2)}")
            ses_all = pd.concat([ses_fit.predict(), ses_fcast])
            ax1.plot(ses_all, ".-", color='r', label = 'ses')
            
            arima_fit = arima(y_to_train)
            arima_fcast = arima_fit.forecast(len(y_to_test))
            mse2 = ((arima_fcast - y_to_test) ** 2).mean()
            print(f"ARIMA RMS Error is {round(np.sqrt(mse2), 2)}")
            arima_all = pd.concat([arima_fit.predict(), arima_fcast])
            ax1.plot(arima_all, ".-", color='b', label = 'arima')

        sarima_fit = sarima(y_to_train,7)
        sarima_fcast = sarima_fit.forecast(len(y_to_test))
        mse2 = ((sarima_fcast - y_to_test) ** 2).mean()
        print(f"SARIMA RMS Error is {round(np.sqrt(mse2), 2)}")
        sarima_all = pd.Series(index=sold_resource.index, data=np.concatenate((sarima_fit.predict(),sarima_fcast)))
        ax1.plot(sarima_all, ".-", color='y', label = 'sarima')
        
        
        ax1.legend(loc = 'best')
        plt.show()
        plt.close(fig=solres_fig)
        plt.close(fig=acf_plot)
        plt.close(fig=pacf_plot)
        plt.close(fig=seasonal_plot)

        # breakpoint()
    



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


