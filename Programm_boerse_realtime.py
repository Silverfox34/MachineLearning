import datetime as dt
import urllib.request
import json
import numpy as np
from datetime import date
import os
import time

def main():
    #begin = dt.date(2022, 12, 1)
    #end = dt.date(2023, 2, 1)


    actualize_files()
    





def actualize_files():
    begin_date = get_last_date_from_files()
    end_date = dt.date.today()
    ticker_symbols = {'AAPL', 'MSFT', 'AMZN', 'ADBE','AAT','AAU',
                    'AB','ABBV','ABC','ABCB'}
    pair_list = create_key_val_pair(ticker_symbols)

    download_data_and_write_to_file(begin_date, end_date, ticker_symbols, pair_list)


def get_last_date_from_files():
    file = open('C:/Users/Moritz/Desktop/Allgemeines/MachineLearning/mydata/'+"AAU"+".txt").read()
    lines = file.split("\n")
    lines = lines[len(lines)-3].split(";")
    lines = lines[0].split("-")

    i0 = int(lines[0])
    i1 = int(lines[1])
    i2 = int(lines[2])
    temp = dt.date(i0, i1, i2)
    temp = temp + dt.timedelta(-1)

    return temp

def download_data_and_write_to_file(begin, end, ticker_symbols, pair_list : dict):
    delta = end - begin
    downloaded_data = np.zeros(shape=[delta.days+1, len(ticker_symbols)])
    downloaded_data[0,:] = list(pair_list.keys())
    counter = 0
    

    

    for item in ticker_symbols:
        dest = 'C:/Users/Moritz/Desktop/Allgemeines/MachineLearning/mydata/'+str(item)+".txt"
        temp = begin

        while temp <= end:
            counter = counter + 1
            try:
                if temp.weekday() != 6 and temp.weekday() != 5:
                    
                    
                    myUrl = 'https://api.polygon.io/v1/open-close/'+str(item)+'/'+str(temp)+'?adjusted=true&apiKey=A9ucsBTluZJyBDw2rZImNSl1sIycyKhd'
                    response = urllib.request.urlopen(myUrl)
                    response_as_string = response.read()
                    jsonObject = json.loads(response_as_string)

                    with open(dest, "a") as myfile:
                        myfile.write(str(temp)+"; "+str(jsonObject.get('open'))+"; "+str(jsonObject.get('close'))+"\r\n")        

                    print(str(temp)+" - "+str(item)+"; SUCCESSFULLY WRITTEN TO FILE")
                    


            except:
                pass
            
            if counter == 5:
                counter = 0
                print("Sleeping")
                time.sleep(70)

                

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