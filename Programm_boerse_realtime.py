import datetime as dt
import urllib.request
import json
import numpy as np
from datetime import date
import os

def main():
    begin = dt.date(2023, 1, 1)
    end = dt.date(2023, 2, 1)
    ticker_symbols = {'AAPL', 'MSFT', 'AMZN', 'ADBE','AAT','AAU',
                        'AB','ABBV','ABC','ABCB'}

    pair_list = create_key_val_pair(ticker_symbols)
    collect_data(begin, end, ticker_symbols, pair_list)

    

def collect_data(begin, end, ticker_symbols, pair_list : dict):
    delta = end - begin
    downloaded_data_dict = np.zeros(shape=[delta.days+1, len(ticker_symbols)])
    downloaded_data_dict[0,:] = list(pair_list.keys())
    
    

    np.savetxt('C:/Users/Moritz/Desktop/Allgemeines/MachineLearning/'+'saved_data.csv', downloaded_data_dict, delimiter=',')
    

    for item in ticker_symbols:
        temp = begin
        while temp <= end:
            
            if temp.weekday() != 6 and temp.weekday() != 0:
                print(temp)
                myUrl = 'https://api.polygon.io/v1/open-close/'+str(item)+'/'+str(temp)+'?adjusted=true&apiKey=A9ucsBTluZJyBDw2rZImNSl1sIycyKhd'
                response = urllib.request.urlopen(myUrl)
                response_as_string = response.read()
                jsonObject : dict= json.loads(response_as_string)
                print(jsonObject.get('close'))
                
            


            temp = temp+dt.timedelta(1)


def create_key_val_pair(ticker_symbols):
    pair_list = {}
    counter = 0
    for item in ticker_symbols:
        pair_list[counter] = str(item)
        counter = counter + 1
    
    return pair_list



if __name__ == "__main__":
    main()