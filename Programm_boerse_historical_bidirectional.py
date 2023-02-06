import datetime as dt

from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import math
from tensorflow import keras
import os
import time
import numpy
from keras.layers import Bidirectional as BD
from keras.layers import LSTM as LSTM


os.environ['KMP_DUPLICATE_LIB_OK']='True'


def main():

    helper_string = 'C:/Users/Moritz/Desktop/Allgemeines/MachineLearning/archive/Stocks_less/'
    onlyfiles = [f for f in listdir(helper_string) if isfile(join(helper_string, f))]

    all_dicts = defaultdict(list)
    actual_data_dict = {}
    bounded_data_dict = defaultdict(list)
    finished_data_dict = {}
    index_keeper = {}
    counter = 0
    counter2 = 0
    counter3 = 0
    counter4 = 0
    delimiter = 20
    pivot = 5
   
    print("Reading dataset....")
    
    for file in onlyfiles:

        paddled_dict = defaultdict(list)
        
        

        begin_date = dt.datetime(2012,12,31)
        end_date = dt.datetime(2016,12,31)

        non_paddled_dict, begin, end = get_date_and_close(helper_string+file)
        #print(begin, end)


        paddled_dict = paddle_data(non_paddled_dict)
        all_dicts[counter] = paddled_dict


        #Here we check if the data is in the timeframe we wanna have
        if(len(paddled_dict.keys()) > 0 and list(paddled_dict.keys())[0] < begin_date and list(paddled_dict.keys())[len(paddled_dict.keys())-1] > end_date):
            #print("Vor 2013 begonnen und nach 2016 geendet: "+file)
            actual_data_dict[counter2] = paddled_dict
            index_keeper[counter2] = file
            counter2 = counter2 + 1

        counter = counter + 1
        

        if file == "advm.us.txt" and False:
            plot_two_dataframes_in_two_graphs(paddled_dict, non_paddled_dict, "Gepaddelter Graph", "Ungepaddelter Graph")


    print("Starting to create bounded data...")
    bounded_data_dict : dict = create_bounded_data_dict(actual_data_dict, begin_date, end_date)
    
    assert_equal_length(bounded_data_dict) 
    new_dict = {}
    [new_begin, bounded_data_dict]= reshape_bounded_data_dict(bounded_data_dict, pivot, begin_date)
    begin_date = new_begin
    
    bounded_data_dict = reorder_bounded_data_dict(bounded_data_dict, pivot, begin_date)
    #print(bounded_data_dict)

    [train_data, train_labels, test_data, test_labels] = create_metadata(bounded_data_dict)
    print(train_data.shape)
    



    #[test_pred, history] = predict_stock_data_bidirectional(train_data, train_labels, test_data, test_labels)
    #predict_stock_data_bidirectional(train_data, train_labels, test_data, test_labels)

    #print(test_labels)
    #print(test_pred)


    #number_of_split_datasets_possible = math.floor(len(bounded_data_dict)/delimiter)


    #FIRST ITERATION
    #second_data_input = start_split_predict(number_of_split_datasets_possible, delimiter, bounded_data_dict)
    

    
    #[train_data, train_labels, test_data, test_labels] = create_metadata(second_data_input)
    #[test_pred, history] = predict_stock_data(train_data, train_labels, test_data, test_labels)



    #compare_results(test_labels, test_pred)


def reorder_bounded_data_dict(bounded_data_dict, pivot : int, begin_date : dt.datetime):
    X_length = len(bounded_data_dict[0])
    Y_length = len(bounded_data_dict)
    mydict = {}
    
    
    if(assert_correct_modulo(bounded_data_dict, pivot) == False):
        print("Data dict is not in shape for pivot " + str(pivot))
        raise Exception

    temp = begin_date

    for i in range(0, Y_length):
        for key in range(0, int(X_length/pivot)):

            time_dict = {}
            for time in range(0, pivot):
               time_dict[time] = bounded_data_dict[i].get(temp)
               temp = temp + dt.timedelta(days=1)
            mydict[key] = time_dict

        bounded_data_dict[i] = mydict

        mydict = {}
        temp = begin_date
    
    return bounded_data_dict
            
        


def assert_correct_modulo(bounded_data_dict, pivot):
    X_length = len(bounded_data_dict[0])
    Y_length = len(bounded_data_dict)

    for k in range(0, Y_length):
        if(len(bounded_data_dict[k]) % pivot != 0):
            return False
    
    return True



def reshape_bounded_data_dict(bounded_data_dict : dict, pivot, begin : dt.datetime):
    X_length = len(bounded_data_dict[0])
    Y_length = len(bounded_data_dict)

    modulo = X_length % pivot
    new_begin = begin + dt.timedelta(days=modulo)
    for k in range(0, Y_length):
        for i in range(0, modulo):
            delta = begin + dt.timedelta(days=i)
            bounded_data_dict[k].pop(delta)

    return [new_begin, bounded_data_dict]


def predict_stock_data_bidirectional(train_data : pd.DataFrame, train_labels, test_data, test_labels):
    train_data_X_length = len(train_data.columns)
    train_data_Y_length = len(train_data.index)

    
    print("Trying to predict stock data...")
    standard_dropout_factor = 0.25

    callback = keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

    model = keras.Sequential()
    model.add(BD(LSTM(units=train_data_X_length, return_sequences=True), merge_mode='concat', input_shape=(train_data_X_length, train_data_Y_length)))
    model.add(BD(LSTM(units=8, activation='relu', return_sequences=True)))
    model.add(BD(LSTM(units=8, activation='relu', return_sequences=True)))
    model.add(BD(LSTM(units=4, activation='relu', return_sequences=True)))
    model.add(keras.layers.Dense(units = 1))


    #model.add(keras.layers.Dense(units = train_data_X_length, input_dim = len(train_data.columns)))
    #model.add(keras.layers.Dense(units = train_data_X_length, activation='relu'))
    #model.add(keras.layers.Dense(units = train_data_X_length, activation='relu'))
    #model.add(keras.layers.Dense(units = train_data_X_length, activation='relu'))
    #model.add(keras.layers.Dense(units = 16, activation='relu'))
    #model.add(keras.layers.Dense(units = 4, activation='relu'))
    #model.add(keras.layers.Dense(units = 1))


    model.compile(loss='mse', optimizer='rmsprop')
    history = model.fit(train_data, train_labels, batch_size=5, epochs = 1000, callbacks=[callback], verbose=1)

    #test_pred = model.predict(test_data)



    #print(test_labels)
    #print(test_pred)

    
    #plot_two_dataframes_in_one_graph(test_labels, test_pred)
    #return [test_pred, history]

def compare_results(test_labels, test_pred):
    print("Prediction and Labels")
    print(test_pred, end='    ')
    print(test_labels)
    
    
def start_split_predict(number_of_split_datasets_possible, delimiter, bounded_data_dict):
    counter4 = 0
    counter3 = 0
    finished_data_dict = {}
    [train_data, train_labels, test_data, test_labels] = create_metadata(bounded_data_dict)
    for i in range(0, number_of_split_datasets_possible-1):
        [test_pred, history] = predict_stock_data(train_data.iloc[: , i*delimiter : (i+1)*delimiter], train_labels, test_data.iloc[: , i*delimiter : (i+1)*delimiter], test_labels)

        if(test_if_bullshit_data(test_pred)):
            finished_data_dict[counter4] = dict(enumerate(test_pred.flatten(), 1))  
            counter4 = counter4 + 1

    
        counter3 = counter3+1
    



    [test_pred, history] = predict_stock_data(train_data.iloc[: , counter3*delimiter : len(bounded_data_dict)-1], train_labels, test_data.iloc[: , counter3*delimiter : len(bounded_data_dict)-1], test_labels)
    if(test_if_bullshit_data(test_pred)):
        finished_data_dict[counter4] = dict(enumerate(test_pred.flatten(), 1))

    second_data_input = pd.DataFrame.from_dict(finished_data_dict, dtype='float')
    return second_data_input
    


def calc_mean_value(mydict, delimiter, length):
    temp = {}

    for i in range(0, len(mydict)):
        dictionary : numpy.ndarray() = mydict[i]
        #print(dictionary) 


def test_if_bullshit_data(mydict):
    length = len(mydict)
    if(mydict[0] == mydict[1] and mydict[0] == mydict[2] and mydict[0] == mydict[3]):
        return False

    return True


def predict_stock_data(train_data : pd.DataFrame, train_labels, test_data, test_labels):
    train_data_X_length = len(train_data.columns)
    
    
    
    print("Trying to predict stock data...")
    standard_dropout_factor = 0.5

    callback = keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

    model = keras.Sequential()

    model.add(keras.layers.Dense(units = train_data_X_length, input_dim = len(train_data.columns)))
    model.add(keras.layers.Dense(units = train_data_X_length, activation='relu'))
    model.add(keras.layers.Dense(units = train_data_X_length, activation='relu'))
    model.add(keras.layers.Dense(units = train_data_X_length, activation='relu'))
    model.add(keras.layers.Dense(units = 16, activation='relu'))
    model.add(keras.layers.Dense(units = 4, activation='relu'))

    model.add(keras.layers.Dense(units = 1))




    #for small datasets:
    #model.add(keras.layers.Dense(units = 16, activation = 'relu'))
    #model.add(keras.layers.Dense(units = 4, activation = 'relu'))
    #model.add(keras.layers.Dense(units = 1))

    model.compile(loss='mse', optimizer='rmsprop')

    history = model.fit(train_data, train_labels, batch_size=30, epochs = 1000,callbacks=[callback], verbose=0)

    test_pred = model.predict(test_data)



    #print(test_labels)
    #print(test_pred)

    
    #plot_two_dataframes_in_one_graph(test_labels, test_pred)
    return [test_pred, history]



def create_metadata(bounded_data_dict):

    data_raw = pd.DataFrame.from_dict(bounded_data_dict, dtype='float')
    train_data_X_length = len(data_raw.columns)
    train_data_Y_length = len(data_raw.index)
    
    dif = 15
    
    train_data = data_raw.iloc[0:train_data_Y_length - dif, :]
    
    train_labels = train_data.iloc[:, 0]
    test_data = data_raw.iloc[train_data_Y_length - dif : train_data_Y_length, :]
    train_data = train_data.iloc[:, 1:data_raw.columns.size]

    test_labels = test_data.iloc[:, 0]
    test_data = test_data.iloc[:, 1:data_raw.columns.size] 


    return [train_data, train_labels, test_data, test_labels]




def plot_two_dataframes_in_two_graphs(list1, list2, descr1, descr2):
    figure, axis = plt.subplots(1, 2)
    

    axis[0].plot(list1.keys(), list1.values())
    axis[0].set_title(descr1)

    axis[1].plot(list2.keys(), list2.values())
    axis[1].set_title(descr2)

    plt.show()


    
def plot_two_dataframes_in_one_graph(list1, list2):

    plt.plot(list1.keys() ,list1, color='r', label='labels')
    plt.plot(list1.keys() ,list2, color='b', label='pred')
    plt.show()


    


def assert_equal_length(bounded_data_dict):
    comparable = len(bounded_data_dict[0])
    for i in range(0,len(bounded_data_dict)):
        if len(bounded_data_dict[i]) != comparable:
            raise ValueError("The dataset does not have the same length for all sets !\nA procedure here would be fatal, so change your spaghetti code to fix that")


def create_bounded_data_dict(actual_data_dict, begin_date, end_date):
    
    bounded_data_dict = {}
    
    temp = begin_date
    counter = 0
    


    for i in range(0,len(actual_data_dict)):
        
        actual_dict = {}
        while True:
            if temp <= end_date:  
                actual_dict[temp] = actual_data_dict[i].get(temp)
                #print(str(temp) +" - "+ str(actual_data_dict[i].get(temp)))
                temp = temp + dt.timedelta(days=1)
            else:
                temp = begin_date
                break
        
        bounded_data_dict[counter] = actual_dict
        counter = counter + 1

    return bounded_data_dict       




def paddle_data(non_paddled_dict: dict):
    paddled_dict = defaultdict(list)


    for entry in non_paddled_dict.keys():
        paddled_dict[entry] = non_paddled_dict[entry]
            
            
        if entry + dt.timedelta(days=1) in non_paddled_dict.keys():
            #print("Next item is here")
            #print("1 Day difference")
            pass
        else:
            if entry == list(non_paddled_dict)[-1]:
                #print("Last item")
                pass
            else:
                iterable = entry
                counter2 = 2
                while True:
                    if entry + dt.timedelta(days=counter2) in non_paddled_dict.keys():
                        for i in range(1, counter2):
                            paddled_dict[entry + dt.timedelta(days=i)] = non_paddled_dict[entry] + (non_paddled_dict[entry + dt.timedelta(days=counter2)]-non_paddled_dict[entry])/counter2*i

                        break

                    else:
                        counter2 = counter2 + 1
    
    return paddled_dict

    
    
        

def get_date_and_close(path):

    begin = 0
    end = 0

    begin_close = True
    end_close = True

    with open(path) as file:
        lines = file.read()

    newdict = defaultdict(list)
    comma_counter=0
    close = ''
    second_row = False;
    date=''

    for i in range(0, len(lines)):

        if(lines[i]=='\n'):
            second_row=True

        if(second_row):

            if lines[i]=='\n': 
                #print(int(lines[i+1:i+5]))
                #print(int(lines[i+6:i+8]))
                #print(int(lines[i+9:i+11]))
                try:
                    date = dt.datetime(int(lines[i+1:i+5]), int(lines[i+6:i+8]), int(lines[i+9:i+11]))
                except:
                    break
                
                #print(lines[i+1:i+11])
                comma_counter=0



            if lines[i]==',':
                comma_counter = comma_counter + 1
            

                if comma_counter==4:
            
                    for j in range(i+1,i+25):
                        if lines[j]==',':
                            break

                        close = close + lines[j]

                    newdict[date] = float(close)

                    if begin_close:
                        begin_close = False
                        begin = newdict[date]

                    end = newdict[date]

                    close = ''
                    date = ''

    return newdict, begin, end       
            



    #plt.plot(x,y)
    #plt.show()
    

if __name__ == "__main__":
    main()
