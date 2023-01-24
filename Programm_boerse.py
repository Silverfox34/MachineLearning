import datetime as dt
from genericpath import exists
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import math
from tensorflow import keras
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


def main():


    #date1 = dt.datetime(2012, 12, 14)
    #date2 = dt.datetime(2012, 11, 29)

    #if date1>date2:
        #print("Date 1 is later than Date 2")
        #dif = date1-date2
        #dif = dif.days
        #print(dif)
    


    helper_string = 'archive/Stocks/'
    onlyfiles = [f for f in listdir(helper_string) if isfile(join(helper_string, f))]

    all_dicts = defaultdict(list)
    actual_data_dict = {}
    bounded_data_dict = defaultdict(list)
    index_keeper = {}
    counter = 0
    counter2 = 0
    max_diff = 0
   
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
        #print(list(actual_data_dict[0].keys())[begin_date])
        #print(actual_data_dict[0].get(begin_date))
        

        if file == "advm.us.txt":
            figure, axis = plt.subplots(1, 2)
            #print(list(paddled_dict.keys())[0])

            axis[0].plot(paddled_dict.keys(), paddled_dict.values())
            axis[0].set_title("Gepaddelter Graph")
            #print(paddled_dict)

            axis[1].plot(non_paddled_dict.keys(), non_paddled_dict.values())
            axis[1].set_title("Ungepaddelter Graph")
            #print(non_paddled_dict)


            #plt.show()
            
            

        counter = counter + 1
    

    print("Starting to create bounded data...")
    bounded_data_dict = create_bounded_data_dict(actual_data_dict, begin_date, end_date)
    assert_equal_length(bounded_data_dict)

    predict_stock_data(bounded_data_dict)
    


def predict_stock_data(bounded_data_dict):
    train_data_length = len(bounded_data_dict)-1
    print("Trying to predict stock data...")
    data_raw = pd.DataFrame.from_dict(bounded_data_dict, dtype='float')
    train_data = data_raw.sample(frac=0.99)
    
    train_labels = train_data.iloc[:, 0]
    test_data = data_raw.drop(train_data.index)


    train_data = train_data.iloc[:, 1:data_raw.columns.size]

    
    test_labels = test_data.iloc[:, 0]
    test_data = test_data.iloc[:, 1:data_raw.columns.size]  
    standard_dropout_factor = 0.5

    callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
    model = keras.Sequential()
    model.add(keras.layers.Dense(units = train_data_length, input_dim = train_data_length))
    model.add(keras.layers.Dense(units = math.ceil(train_data_length), activation='sigmoid'))
    model.add(keras.layers.Dropout(standard_dropout_factor))
    model.add(keras.layers.Dense(units = math.ceil(train_data_length), activation='sigmoid'))
    model.add(keras.layers.Dropout(standard_dropout_factor))
    model.add(keras.layers.Dense(units = math.ceil(train_data_length), activation='sigmoid'))
    model.add(keras.layers.Dropout(standard_dropout_factor))
    model.add(keras.layers.Dense(units = math.ceil(train_data_length), activation='sigmoid'))
    model.add(keras.layers.Dropout(standard_dropout_factor))
    model.add(keras.layers.Dense(units = math.ceil(train_data_length), activation='sigmoid'))
    model.add(keras.layers.Dropout(standard_dropout_factor))
    model.add(keras.layers.Dense(units = math.ceil(train_data_length), activation='sigmoid'))
    model.add(keras.layers.Dropout(standard_dropout_factor))
    model.add(keras.layers.Dense(units = math.ceil(train_data_length), activation='sigmoid'))
    model.add(keras.layers.Dropout(standard_dropout_factor))
    model.add(keras.layers.Dense(units = math.ceil(train_data_length), activation='sigmoid'))
    model.add(keras.layers.Dropout(standard_dropout_factor))
    model.add(keras.layers.Dense(units = math.ceil(train_data_length), activation='sigmoid'))
    model.add(keras.layers.Dropout(standard_dropout_factor))
    model.add(keras.layers.Dense(units = math.ceil(train_data_length), activation='sigmoid'))
    model.add(keras.layers.Dropout(standard_dropout_factor))
    model.add(keras.layers.Dense(units = math.ceil(train_data_length), activation='sigmoid'))
    model.add(keras.layers.Dropout(standard_dropout_factor))
    model.add(keras.layers.Dense(units = math.ceil(train_data_length), activation='sigmoid'))
    model.add(keras.layers.Dropout(standard_dropout_factor))
    model.add(keras.layers.Dense(units = math.ceil(train_data_length/2), activation='sigmoid'))
    model.add(keras.layers.Dropout(standard_dropout_factor))
    model.add(keras.layers.Dense(units = math.ceil(train_data_length/4), activation='sigmoid'))
    model.add(keras.layers.Dropout(standard_dropout_factor))
    model.add(keras.layers.Dense(units = math.ceil(train_data_length/8), activation='sigmoid'))
    model.add(keras.layers.Dropout(standard_dropout_factor))
    model.add(keras.layers.Dense(units = math.ceil(train_data_length/16), activation='sigmoid'))
    model.add(keras.layers.Dropout(standard_dropout_factor))
    model.add(keras.layers.Dense(units = math.ceil(train_data_length/32), activation='sigmoid'))
    model.add(keras.layers.Dropout(standard_dropout_factor))
    model.add(keras.layers.Dense(units = math.ceil(train_data_length/64), activation='sigmoid'))
    model.add(keras.layers.Dropout(standard_dropout_factor))
    model.add(keras.layers.Dense(units = math.ceil(train_data_length/128), activation='sigmoid'))
    model.add(keras.layers.Dropout(standard_dropout_factor))
    model.add(keras.layers.Dense(units = math.ceil(train_data_length/256), activation='sigmoid'))
    model.add(keras.layers.Dropout(standard_dropout_factor))

    #for small datasets:
    #model.add(keras.layers.Dense(units = 16, activation = 'relu'))
    #model.add(keras.layers.Dense(units = 4, activation = 'relu'))
    #model.add(keras.layers.Dense(units = 1))


    
    model.add(keras.layers.Dense(units = 1))

    model.compile(loss='mse', optimizer='rmsprop')

    history = model.fit(train_data, train_labels, epochs = 50, callbacks=[callback])

    test_pred = model.predict(test_data)



    print(test_labels)
    print(test_pred)


    
    

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
