import pandas as pd
import numpy as np
import io

from datetime import datetime
from datetime import timedelta

TTL = 7

def dateparse(d,t):
    dt = d + " " + t
    return pd.to_datetime(dt, format='%Y-%m-%d %I:%M %p')

def read_data():
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
    return purchased, sold, shelf_life

def calculate_waste(purchased, sold, shelf_life):
    wasted = []
    wasted_pd = pd.DataFrame(columns = ['resource_id','resource_name','unit','wasted','date','time'])

    purchased.sort_values(by='datetime', inplace=True)
    sold.sort_values(by='datetime', inplace=True)

    # für jeden kauf checken wie viel in den nächsten TTL tagen verkauft wurde
    # verkäufe werden nur 1x gezählt
    # ACHTUNG: units werden immer als KG angenommen
    for i, purchase_row in purchased.iterrows():
        if np.isnan(purchase_row['purchased']):
            continue
        TTL = shelf_life[shelf_life['resource_name'] == purchase_row['resource_name']]['shelf_life_days'].item()
        
        ### Abfalltag
        wasted_date = purchase_row['datetime'] + timedelta(days=TTL)
        
        print ("---")
        print ("Waste report for Product: " + purchase_row['resource_name'] + " delivered on " + str(purchase_row['datetime']))
        print ("Quantity: " + str(purchase_row['purchased']) + " kgs")
        print ("Average shelf life: " + str(TTL) + " days")
        print ("Should all be sold by: " + str(wasted_date) + "\n")
        
        ### Nächster Einkaufstag      
        next_purchased_date  = purchased.loc[(purchased['resource_name'] == purchase_row['resource_name']) & (purchased['datetime'] > purchase_row['datetime'])]['datetime'].min()
        
        # Calculate all that is sold before a next purchase
        mask_beforePurchase = (sold['datetime'] >= purchase_row['datetime']) & (sold['datetime'] < next_purchased_date) & (sold['resource_name'] == purchase_row['resource_name'])

        sold_beforePurchase = sold.loc[mask_beforePurchase]['sold'].sum()

        if sold_beforePurchase == purchase_row['purchased']:
            print ("\nBravo. Zero waste on this delivery.\n")
        elif sold_beforePurchase > purchase_row['purchased']:
            print ("\nWe have sold more than we had, possibly there was some goods in storage.\n")
        else:
            # we have sold less than it was purchased before the next purchase
            # breakpoint()
            if wasted_date < next_purchased_date:
                # there is no effect on subsequent sales,
                # the goods is wasted before next purchase
                wasted_sum = purchase_row['purchased'] - sold_beforePurchase
                # we assume that all that is sold before next purchase
                # comes from the previous purchase, even if that might be
                # beyond TTL, as the goods must come from somewhere
                print (f"Sellings total: {sold_beforePurchase} kgs were sold.")
                print (f"Calculated waste: {wasted_sum} on: {wasted_date}\n")
            else:
                # we still can sell something after the next purchase
                mask_beforeWaste = (sold['datetime'] >= purchase_row['datetime']) & (sold['datetime'] < wasted_date) & (sold['resource_name'] == purchase_row['resource_name'])
                sold_beforeWaste = sold.loc[mask_beforeWaste]['sold'].sum()
                wasted_sum = purchase_row['purchased'] - sold_beforeWaste
                if wasted_sum <= 0:
                    print ("\nBravo. Zero waste on this delivery.\n")
                else:
                    print (f"Sellings total: {sold_beforeWaste} kgs were sold.")
                    print (f"Calculated waste: {wasted_sum} on: {wasted_date}\n")
                # calculate what to detract from sales in order to take into account
                # the sale of old goods
                mask_betwPurchasedWaste = (sold['datetime'] >= next_purchased_date) & (sold['datetime'] < wasted_date) & (sold['resource_name'] == purchase_row['resource_name'])
                if sum(mask_betwPurchasedWaste) > 0:
                    # detract old goods sales from sales
                    # sales of old goods after next purchase are at max the remaining quantity from the previous purchase
                    sold_betwPurchasedWaste = min(sold.loc[mask_betwPurchasedWaste]['sold'].sum(), purchase_row['purchased'] - sold_beforePurchase)
                    # remove equally from sales, this is not correct as some sales can go negative, but it should not matter when you sum sales
                    sold.loc[mask_betwPurchasedWaste, 'sold'] -= sold_betwPurchasedWaste / sold.loc[mask_betwPurchasedWaste, 'sold'].count()

            if wasted_sum > 0:
                wasted.append({'resource_id': purchase_row['resource_id'], 
                                            'resource_name': purchase_row['resource_name'],
                                            'unit': purchase_row['unit'],
                                            'wasted': wasted_sum,
                                            'date':  wasted_date.strftime('%Y-%m-%d'),
                                            'time':  wasted_date.strftime('%I:%M %p')
                                            })
                print ("Waste event logged.\n")

    wasted_pd = pd.DataFrame(wasted)    
    wasted_pd.to_csv('wasted_mod.csv', index=False)

def main():
    purchased, sold, shelf_life = read_data()
    calculate_waste(purchased, sold, shelf_life)

if __name__ == "__main__":
    main()