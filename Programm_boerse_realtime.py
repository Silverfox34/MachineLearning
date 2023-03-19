import datetime as dt
import urllib.request
import json
import numpy as np
import tensorflow as tf
from keras.layers.core import Dense, Dropout, Flatten
import keras
import sys
import math
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import normalize
from UsefulFunctions import HistoryPlotter as HP

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def main():
    np.set_printoptions(threshold=sys.maxsize)
    begin = dt.date(2022, 12, 10)
    end = dt.date(2023, 2, 10)
    vector_size = 8
    normalize_flag = False

  
    ticker_symbols = ['AAPL', 'MSFT', 'AMZN', 'ADBE','AAT','AAU', 'AB','ABBV','ABC','ABCB']
    #ticker_symbols = ['AAPL', 'MSFT', 'AMZN', 'ADBE', 'AAT']
    pair_list = create_key_val_pair(ticker_symbols)

    #actualize_files(ticker_symbols, pair_list)
    [morning_numpy_array, evening_numpy_array] = read_files(ticker_symbols, begin, end)
    sequenced_dataset_morning = create_sequence_dataset(morning_numpy_array, vector_size, normalize_flag)
    sequenced_dataset_evening = create_sequence_dataset(evening_numpy_array, vector_size, normalize_flag)
    
    
    
    [X_train, X_test, Y_train, Y_test] = create_train_test_split(sequenced_dataset_morning, ticker_symbols)
    [X_train, X_test] = reshape_data_for_nn(X_train, X_test, normalize_flag)
    create_neural_net_and_feed_it_yummy_yummy(X_train, X_test, Y_train, Y_test, vector_size)





def create_neural_net_and_feed_it_yummy_yummy(X_train : np.array,  X_test : np.array, Y_train : np.array, Y_test : np.array, vector_size):
    stock_amount = X_train.shape[0]
    data_length = X_train.shape[1]   
    dropout_rate = 0.3
    #log_base = 2
    
    layer_size = 6
    layer_number = 6

    
    early_stopping_callback = keras.callbacks.EarlyStopping(monitor='loss', patience=30, restore_best_weights=True)
    

    model = keras.Sequential()


    model.add(Dense(units = data_length, input_dim = data_length))
    
    for i in range(0, layer_number):
        model.add(Dense(units = layer_size, activation='sigmoid'))
        model.add(Dropout(rate = dropout_rate))
    
    model.add(Dense(units = vector_size))

    
    model.compile(loss='mse', optimizer='sgd', metrics='mean_absolute_error')
    
    history = model.fit(x=X_train, y=Y_train, batch_size=1, validation_data=(X_test, Y_test), epochs=200, callbacks=[early_stopping_callback])

    figure, axis = plt.subplots(5, 2)
    my_predictions = []
    
    
    for i in range(0, Y_test.shape[0]):
        #print("Output as it should look like: ")
        #print(Y_test[i])
        #print("-----------------------------------")
        X_test_reshaped_line = np.reshape(a=X_test[i], newshape=(1, X_test[i].shape[0]))

        print("X_test[i] shape: ")
        print(X_test_reshaped_line.shape)
        print("-----------------------------")
        my_pred = model.predict(X_test_reshaped_line)
        my_predictions.append(my_pred)
        print("Prediction: ")
        print(my_pred.shape)

        axis[i, 0].plot(Y_test[i])
        axis[i, 0].set_title("Test-set values")
        
        axis[i, 1].plot(my_predictions[i][0])
        axis[i, 1].set_title("Predicted values")

        print("############################################\n############################################")

    plt.show()



def reshape_data_for_nn(X_train : np.array, X_test : np.array, normalize_flag):

    if normalize_flag:
        X_train = np.reshape(a=X_train, newshape=(X_train.shape[0], X_train.shape[1]*X_train.shape[3]))
        X_test = np.reshape(a=X_test, newshape=(X_test.shape[0], X_test.shape[1]*X_test.shape[3]))
    else:
        X_train = np.reshape(a=X_train, newshape=(X_train.shape[0], X_train.shape[1]*X_train.shape[2]))
        X_test = np.reshape(a=X_test, newshape=(X_test.shape[0], X_test.shape[1]*X_test.shape[2]))
    
    return [X_train, X_test]
    


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

    



def create_sequence_dataset(numpy_array : np.array, seq_length : int, normalize_flag):
    sequence_list = []
    big_sequence_list = []
    
    
    for i in range(0, numpy_array.shape[0]-seq_length):

        for j in range(0, numpy_array.shape[1]):
            temp = numpy_array[i : i+seq_length, j]

            if(normalize_flag):
                temp = normalize([temp])


            sequence_list.append(temp)
        
        
        
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
    #begin_date = get_last_date_from_files()
    begin_date = dt.date(2018, 4, 1)
    end_date = dt.date.today()
    #end_date = get_first_date_from_files()
    download_data_and_write_to_file(begin_date, end_date, ticker_symbols, pair_list)

def get_first_date_from_files():
    file = open('C:/Users/Moritz/Desktop/Allgemeines/MachineLearning/mydata/'+"AAU"+".txt").read()
    lines = file.split("\n")
    lines = lines[0].split(";")
    lines = lines[0].split("-")

    i0 = int(lines[0])
    i1 = int(lines[1])
    i2 = int(lines[2])
    temp = dt.date(i0, i1, i2)
    temp = temp + dt.timedelta(-1)

    return temp


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
    #counter = 0


    for item in ticker_symbols:
        dest = 'C:/Users/Moritz/Desktop/Allgemeines/MachineLearning/mydata/'+str(item)+".txt"
        temp = begin
        api_key = 'msEs_vaY1U3zMeJz3dWpnalk16rzWNze'

        while temp <= end:
            #counter = counter + 1
            try:
                if temp.weekday() != 6 and temp.weekday() != 5:

                    
                    
                    myUrl = 'https://api.polygon.io/v1/open-close/'+str(item)+'/'+str(temp)+'?adjusted=true&apiKey='+api_key
                    response = urllib.request.urlopen(myUrl)
                    response_as_string = response.read()
                    jsonObject = json.loads(response_as_string)

                    with open(dest, "a") as myfile:
                        myfile.write(str(temp)+"; "+str(jsonObject.get('open'))+"; "+str(jsonObject.get('close'))+"\r\n")        

                    print(str(temp)+" - "+str(item)+"; SUCCESSFULLY WRITTEN TO FILE")
                    


            except:
                pass
            
            #if counter == 5:
                #counter = 0
                #print("Sleeping")
                #time.sleep(70)

                

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