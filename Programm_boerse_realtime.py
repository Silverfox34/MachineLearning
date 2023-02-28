import datetime as dt
import urllib.request
import json
import numpy as np
from datetime import date
import os
import time
from keras.layers import Dense, Dropout
import keras
from sklearn.model_selection import train_test_split
import pandas as pd
import sys

def main():
    np.set_printoptions(threshold=sys.maxsize)
    begin = dt.date(2022, 12, 10)
    end = dt.date(2023, 2, 10)
    vector_size = 5
  
    ticker_symbols = ['AAPL', 'MSFT', 'AMZN', 'ADBE','AAT','AAU', 'AB','ABBV','ABC','ABCB']
    #ticker_symbols = ['AAPL', 'MSFT', 'AMZN', 'ADBE', 'AAT']
    #pair_list = create_key_val_pair(ticker_symbols)

    #actualize_files(ticker_symbols, pair_list)
    [morning_numpy_array, evening_numpy_array] = read_files(ticker_symbols, begin, end)
    sequenced_dataset_morning = create_sequence_dataset(morning_numpy_array, vector_size)
    sequenced_dataset_evening = create_sequence_dataset(evening_numpy_array, vector_size)
    

    [X_train, X_test, Y_train, Y_test] = create_train_test_split(sequenced_dataset_morning, ticker_symbols)
    
    create_neural_net_and_feed_it_yummy_yummy(X_train, X_test, Y_train, Y_test)





def create_train_test_split(numpy_dataset : np.array, ticker_symbols : list):  
    split_var = -5
    target : np.array = numpy_dataset[:, 0, :]
    input : np.array = numpy_dataset[:, 1:, :]
    
    
    Y_train = target[:split_var]
    Y_test = target[target.shape[0] + split_var : target.shape[0]]

    #for i in range(0, input.shape[0]):
        #temp_list_train.append(input[i][:split_var])
        #temp_list_test.append(input[i][input.shape[0] + split_var : input.shape[0]]) 

    X_train = input[:split_var, :, :]
    X_test = input[input.shape[0] + split_var : input.shape[0], :, :]
    

    return [X_train, X_test, Y_train, Y_test]

    





def create_neural_net_and_feed_it_yummy_yummy(X_train : np.array,  X_test : np.array, Y_train : np.array, Y_test : np.array):
    stock_amount = X_train.shape[0]
    vectors_amount = X_train.shape[1]
    time_steps = X_train.shape[2]
    dropout_rate = 0.4

    early_stopping_callback = keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

    model = keras.Sequential()
    model.add(Dense(units=time_steps, input_shape=(vectors_amount, time_steps)))

    for i in range(0, time_steps-2):
        model.add(Dense(units=time_steps, activation='relu'))
        model.add(Dropout(rate=dropout_rate))

    model.compile(loss='mse', optimizer='adam')
    
    history = model.fit(x=X_train, y=Y_train, batch_size=time_steps, validation_data=(X_test, Y_test), epochs = 1000, callbacks=[early_stopping_callback])
    
    




def create_sequence_dataset(numpy_array : np.array, seq_length : int):
    sequence_list = []
    big_sequence_list = []
    
    
    for i in range(0, numpy_array.shape[0]-seq_length):

        for j in range(0, numpy_array.shape[1]):
            sequence_list.append(numpy_array[i : i+seq_length, j])
        
        

        
        big_sequence_list.append(np.array(sequence_list, dtype='float'))
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
        #morning_list.append(str(item))
        #evening_list.append(str(item))

        for line in lines:
            
            data = line.split(";")

            if data[0] == '' or data[1] == '' or data[2] == '':
                continue

            morning_list.append(data[1].replace(" ",""))
            evening_list.append(data[2].replace(" ",""))
        
        morning_numpy_array.append(np.array(morning_list))
        evening_numpy_array.append(np.array(evening_list))

        counter = counter + 1

    return [np.transpose(np.array(morning_numpy_array)), np.transpose(np.array(evening_numpy_array))]





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