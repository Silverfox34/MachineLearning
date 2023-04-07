import datetime as dt
import urllib.request
import json
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Dropout, LSTM, Embedding
import keras
import sys
import math
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import normalize
from keras.losses import MeanSquaredError as MSE
from keras.losses import CategoricalHinge as CH
from keras.losses import Poisson as PS

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def main():
    np.set_printoptions(threshold=sys.maxsize)
    begin = dt.date(2022, 12, 10)
    end = dt.date(2023, 2, 10)
    vector_size = 16
    normalize_flag = False

  
    ticker_symbols = ['AAPL', 'MSFT', 'AMZN', 'ADBE','AAT','AAU', 'AB','ABBV','ABC','ABCB']
    pair_list = create_key_val_pair(ticker_symbols)

    #actualize_files(ticker_symbols, pair_list)
    [morning_numpy_array, evening_numpy_array] = read_files(ticker_symbols, begin, end)

    
    
    [X_train, X_test, Y_train, Y_test] = create_train_test_split_LSTM(morning_numpy_array, ticker_symbols)
    
    
    
    create_neural_net_and_feed_it_yummy_yummy(X_train, X_test, Y_train, Y_test, vector_size)
    return
    
    
    sequenced_dataset_morning = create_sequence_dataset(morning_numpy_array, vector_size, normalize_flag)
    sequenced_dataset_evening = create_sequence_dataset(evening_numpy_array, vector_size, normalize_flag)
    
    
    
    [X_train, X_test, Y_train, Y_test] = create_train_test_split(sequenced_dataset_morning, ticker_symbols)
    [X_train, X_test] = reshape_data_for_nn(X_train, X_test, normalize_flag)
    create_neural_net_and_feed_it_yummy_yummy(X_train, X_test, Y_train, Y_test, vector_size)



def create_neural_net_and_feed_it_yummy_yummy(X_train : np.array,  X_test : np.array, Y_train : np.array, Y_test : np.array, vector_size):

    dropout_rate = 0.3
    layer_size = 8
    layer_number = 3
    batch_size_value = 10
    tensor_var = 32
    data_vectors = X_train.shape[1]


    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_test.shape)
    print("-----------------")
    early_stopping_callback = keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    
    model = keras.Sequential()
    
    model.add(Dense(units = data_vectors, input_shape=(batch_size_value, data_vectors)))

    model.add(LSTM(units = data_vectors, activation='relu', return_sequences=True, input_shape=[batch_size_value, data_vectors]))
    
    for i in range(0, layer_number):
        model.add(LSTM(units = data_vectors, activation='relu', return_sequences=True, input_shape=[batch_size_value, data_vectors]))
        
    model.add(Dense(units = data_vectors, activation='relu'))
    model.add(Dense(units = 1))

    #kl_divergence, categorical_hinge has different results
    #log_cosh was best
    model.compile(loss="mse", optimizer='adam')
    
    history = model.fit(x=X_train, y=Y_train, validation_data=(X_test, Y_test), epochs=50, callbacks=[early_stopping_callback], batch_size=batch_size_value)

    figure, axis = plt.subplots(5, 2)
    my_predictions = []
    

   

    
    
    for i in range(0, Y_test.shape[0]):
        print("Output as it should look like: ")
        print(Y_test[i])
        print("-----------------------------------")
        X_test_reshaped_line = np.reshape(a=X_test[i], newshape=(1, X_test[i].shape[0]))

        my_pred = model.predict(X_test_reshaped_line)
        my_predictions.append(my_pred)
        print("Prediction: ")
        print(my_pred)

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



def create_train_test_split_LSTM(numpy_dataset : np.array, ticker_symbols : list):  
    split_var = -17
    target : np.array = numpy_dataset[:][0]
    input : np.array = numpy_dataset[:][1:]
    
    
    
    Y_train = target[:split_var]
    Y_test = target[target.shape[0] + split_var : target.shape[0]]

   
    X_train : np.array = input[:, :split_var]  
    X_test = input[:, input.shape[1] + split_var : input.shape[1]]

    #reshaping
    Y_train = Y_train.reshape(1, Y_train.shape[0])
    Y_test = Y_test.reshape(1, Y_test.shape[0])

    X_train = np.transpose(X_train)
    Y_train = np.transpose(Y_train)
    X_test = np.transpose(X_test)
    Y_test = np.transpose(Y_test)

    #X_train = X_train.reshape(1, X_train.shape[0], X_train.shape[1])
    #Y_train = Y_train.reshape(1, Y_train.shape[0], Y_train.shape[1])
    #X_test = X_test.reshape( X_test.shape[0], X_test.shape[1])
    #Y_test = Y_test.reshape( Y_test.shape[0], Y_test.shape[1])
    

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
    morning_numpy_array = []
    evening_numpy_array = []
    
    
    for item in ticker_symbols:
        file = open('C:/Users/Moritz/Desktop/Allgemeines/MachineLearning/mydata/'+str(item)+".txt").read()
        lines = file.split("\n")

        morning_list = []
        evening_list = []

        for line in lines:
            if len(line) < 2:
                continue

            data = line.split(";")

            if data[0] == '' or data[1] == '' or data[2] == '' or data[0].replace(" ","") == 'None' or data[1].replace(" ","") == 'None' or data[2].replace(" ","") == 'None':
                continue
            
            morning_list.append(float(data[1].replace(" ","")))
            evening_list.append(float(data[2].replace(" ","")))

       
        morning_numpy_array.append(morning_list)
        evening_numpy_array.append(evening_list)

        counter = counter + 1

    #return [np.transpose(np.array(morning_numpy_array, dtype='object')), np.transpose(np.array(evening_numpy_array, dtype='object'))]
    return [np.array(morning_numpy_array), np.array(evening_numpy_array)]

    





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