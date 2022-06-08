import pandas as pd
import numpy as np
import io

from datetime import datetime
from datetime import timedelta

TTL = 7

def dateparse(d,t):
    dt = d + " " + t
    return pd.to_datetime(dt, format='%Y-%m-%d %I:%M %p')

purchased = pd.read_csv(r"purchased.csv", 

                           sep = ',',

                           dtype={'resource_id':'string',
                                  'resource_name':'string',
                                  'unit':'string',
                                  'purchased':'float64'
                                 },
                            
                        parse_dates={'datetime': ['date', 'time']}, 
                            
                        date_parser=dateparse)

sold = pd.read_csv(r"sold.csv", 

                           sep = ',',

                           dtype={'resource_id':'string',
                                  'resource_name':'string',
                                  'unit':'string',
                                  'sold':'float64'
                                 },
                            
                        parse_dates={'datetime': ['date', 'time']}, 
                            
                        date_parser=dateparse)

shelf_life = pd.read_csv(r"shelf_life.csv", 

                           sep = ';',

                           dtype={'resource_name':'string',
                                  'shelf_life_days':'int16'
                                 }
                        )

wasted = pd.DataFrame(columns = ['resource_id','resource_name','unit','wasted','date','time'])

purchased.sort_values(by='datetime', inplace=True)
sold.sort_values(by='datetime', inplace=True)

# für jeden kauf checken wie viel in den nächsten TTL tagen verkauft wurde
# verkäufe werden nur 1x gezählt
# ACHTUNG: units werden immer als KG angenommen
for i, purchase_row in purchased.iterrows():
    
    TTL = shelf_life[shelf_life['resource_name'] == purchase_row['resource_name']]['shelf_life_days'].item()
    
    ### Abfalltag
    wasted_date = purchase_row['datetime'] + timedelta(days=TTL)
    
    ### Nächster Einkaufstag      
    next_purchased_date  = purchased.loc[(purchased['resource_name'] == purchase_row['resource_name']) & (purchased['datetime'] > purchase_row['datetime'])]['datetime'].min()
    
    ### wenn es einen nächsten kaufevent gibt
    after_ttl_sellings = 0
    mask = (sold['datetime'] >= purchase_row['datetime']) & (sold['datetime'] <= wasted_date) & (sold['resource_name'] == purchase_row['resource_name'])
    
    print ("---")
    print ("Waste report for Product: " + purchase_row['resource_name'] + " delivered on " + str(purchase_row['datetime']))
    print ("Quantity: " + str(purchase_row['purchased']) + " kgs")
    print ("Average shelf life: " + str(TTL) + " days")
    print ("Should all be sold by: " + str(wasted_date) + "\n")
        
    if pd.isnull(next_purchased_date):
        after_ttl_sellings = 1
    else:
        if next_purchased_date > wasted_date:
            after_ttl_sellings = 2
            mask = (sold['datetime'] >= purchase_row['datetime']) & (sold['datetime'] <= next_purchased_date) & (sold['resource_name'] == purchase_row['resource_name'])
    
    sold_sum = sold.loc[mask]['sold'].sum()
    wasted_sum = purchase_row['purchased'] - sold_sum
    
    #### TODO: checken warum manche wasted_sums NAN sind ....
    if np.isnan(sold_sum):
        sold_sum = 0

    if np.isnan(wasted_sum):
        wasted_sum = 0
        
    match after_ttl_sellings:
        case 2:
            print("The data shows sellings after assumed shelf life which should be taken into account calculating the waste.")
            print("All sellings until next purchase will be taken into account: " + str(next_purchased_date))
            print("Thats " + str (next_purchased_date - wasted_date) + " after the expected shelf life.")
            print("If that seems unrealistic maybe shelf life should be adjusted ...\n")
    
    if sold_sum > 0:
        print ("List of sellings taken into account between: " + str(min(sold.loc[mask]['datetime'])) + " and " + str(max(sold.loc[mask]['datetime'])) + ":\n")
        print (sold.loc[mask])
        print ("\n")
    
    print ("Sellings total: " + str(sold_sum) + " kgs were sold.")
    print ("Calculated waste: " + str(wasted_sum) + " on: " + str(wasted_date) + "\n")

    # wenn waste in den minusbreich rutscht, sollten die verkäufe bestehen bleiben
    if wasted_sum == 0:
            print ("\nBravo. Zero waste on this delivery.\n")
    else:
        if wasted_sum < 0:

            #dann gibt es kein wasted, aber die differenz muss in den verkäufen bestehen bleiben
            #es müssen die soviele verkäufe im dataset bleiben dass es bei 0 bleibt

            #### Den "Überverkauf" wieder gleichmäßig verteilt ....
            breakpoint()
            sold.loc[mask, 'sold'] = (wasted_sum * -1) / sold.loc[mask, 'sold'].count()

            #### alle verkaufsevents entfernen bis auf das letzte das bekommt wasted_sum * -1
            #sold.loc[mask & (sold['datetime'] == max(sold.loc[mask]['datetime'])), 'sold'] = wasted_sum * -1
            print ("The data shows that more was sold then purchased. So we dont throw anything away actually.")
            print ("We redistribute the sellings amount below zero evenly on the original selling dates.")
            print ("So they can be taken into account calculating the waste of a potential later delivery.\n")
            print (sold.loc[mask])
            print ("\nBravo. Zero waste on this delivery.\n")

        else:
            # alle abgerechneten verkäufe rausnehmen
            sold = sold.drop(sold[mask].index)
            # dataframe mit den verschwendeten lebensmitteln wird aufgebaut

            # 5 Tage nach dem letzten aufgezeichneten Verkauf werden keine Waste Dates mehr generiert ....
            if wasted_date <= sold['datetime'].max() + timedelta(days=5):
                wasted = wasted.append({'resource_id': purchase_row['resource_id'], 
                                        'resource_name': purchase_row['resource_name'],
                                        'unit': purchase_row['unit'],
                                        'wasted': wasted_sum,
                                        'date':  wasted_date.strftime('%Y-%m-%d'),
                                        'time':  wasted_date.strftime('%I:%M %p')
                                        }, ignore_index=True)
                print ("Waste event logged.\n")
            else:
                print ("The calcualted waste event is to far in the future to be reliable. It will not be logged.")
                print ("Maybe there is potential for a predictive approach ...\n")
            
    print ("---\n\n")

wasted.to_csv('wasted.csv', index=False)