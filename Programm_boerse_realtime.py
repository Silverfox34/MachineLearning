import datetime as dt
import urllib.request
import json
import numpy as np
from datetime import date
import os
import time
from keras.layers import Dense
import keras

def main():
    begin = dt.date(2022, 12, 10)
    end = dt.date(2023, 2, 10)
    vector_size = 5

    ticker_symbols = ['AAPL', 'MSFT', 'AMZN', 'ADBE','AAT','AAU',
                    'AB','ABBV','ABC','ABCB']
    pair_list = create_key_val_pair(ticker_symbols)

    #actualize_files(ticker_symbols, pair_list)
    [morning_numpy_array, evening_numpy_array] = read_files(ticker_symbols, begin, end)
    sequenced_dataset_morning = create_sequence_dataset(morning_numpy_array, vector_size)
    sequenced_dataset_evening = create_sequence_dataset(evening_numpy_array, vector_size)

    create_neural_net_and_feed_it_yummy_yummy(sequenced_dataset_morning)

def create_neural_net_and_feed_it_yummy_yummy(dataset : np.array):
    stock_amount = dataset.shape[0]
    vectors_amount = dataset.shape[1]
    time_steps = dataset.shape[2]
    model = keras.Sequential()

   
    
    


def create_sequence_dataset(numpy_array : np.array,seq_length : int):
    sequence_list = []
    big_sequence_list = []
    mydict = {}
    

    for i in range(0, numpy_array.shape[0]):
        #sequence_list.append(numpy_array[i][0])

        for j in range(1, numpy_array.shape[1]-1-seq_length):
            sequence_list.append(numpy_array[i][j : j+seq_length])

       
        
        big_sequence_list.append(np.array(sequence_list))
        sequence_list = []

    return np.array(big_sequence_list)
        

    


def read_files(ticker_symbols, begin_date, end_date):
    delta = end_date-begin_date
    delta = delta.days
    
    counter = 0
    #morning_numpy_array = np.empty([len(ticker_symbols), int(delta)])
    #evening_numpy_array = np.empty([len(ticker_symbols), int(delta)])
    morning_numpy_array = []
    evening_numpy_array = []

    for item in ticker_symbols:
        file = open('C:/Users/Moritz/Desktop/Allgemeines/MachineLearning/mydata/'+str(item)+".txt").read()
        lines = file.split("\n")

        morning_list = []
        evening_list = []
        morning_list.append(str(item))
        evening_list.append(str(item))

        for line in lines:
            
            data = line.split(";")

            if data[0] == '' or data[1] == '' or data[2] == '':
                continue

            morning_list.append(data[1].replace(" ",""))
            evening_list.append(data[2].replace(" ",""))
        
        morning_numpy_array.append(np.array(morning_list))
        evening_numpy_array.append(np.array(evening_list))

        counter = counter + 1

    return [np.array(morning_numpy_array), np.array(evening_numpy_array)]





def actualize_files(ticker_symbols, pair_list):
    begin_date = get_last_date_from_files()
    end_date = dt.date.today()
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
    temp = temp + dt.timedelta(1)

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