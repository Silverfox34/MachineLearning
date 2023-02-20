import numpy as np
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers import LSTM as LSTM, Dense, Dropout
from keras import callbacks
import tensorflow as tf
import matplotlib.pyplot as plt
from UsefulFunctions import HistoryPlotter as hp
from UsefulFunctions import StockDataLoader as sdl
import datetime as dt

def main():
    time_steps = 3
    n = 4

    m=42
    

    t = 3*n
    
    my_path = 'C:/Users/Moritz/Desktop/Allgemeines/MachineLearning/archive/Stocks_less/'
    begin_date = dt.datetime(2012,12,31)
    end_date = dt.datetime(2016,12,31)

    
    data2 = np.array([
        
                    [[0,0,0,0],[0,0,0,0],[t,t,t,t],[m,m,m,m],[m,m,m,m],[m,m,m,m]],
                    [[0,0,0,0],[0,0,0,0],[t,t,t,t],[m,m,m,m],[m,m,m,m],[m,m,m,m]],
                    [[0,0,0,0],[0,0,0,0],[t,t,t,t],[m,m,m,m],[m,m,m,m],[m,m,m,m]],
                    [[0,0,0,0],[0,0,0,0],[t,t,t,t],[m,m,m,m],[m,m,m,m],[m,m,m,m]],
                    [[0,0,0,0],[0,0,0,0],[t,t,t,t],[m,m,m,m],[m,m,m,m],[m,m,m,m]],
                    [[0,0,0,0],[0,0,0,0],[t,t,t,t],[m,m,m,m],[m,m,m,m],[m,m,m,m]],
                    [[0,0,0,0],[0,0,0,0],[t,t,t,t],[m,m,m,m],[m,m,m,m],[m,m,m,m]],
                    [[0,0,0,0],[0,0,0,0],[t,t,t,t],[m,m,m,m],[m,m,m,m],[m,m,m,m]],
                    [[0,0,0,0],[0,0,0,0],[t,t,t,t],[m,m,m,m],[m,m,m,m],[m,m,m,m]],
                    [[0,0,0,0],[0,0,0,0],[t,t,t,t],[m,m,m,m],[m,m,m,m],[m,m,m,m]],
                    [[0,0,0,0],[0,0,0,0],[t,t,t,t],[m,m,m,m],[m,m,m,m],[m,m,m,m]],
                    [[0,0,0,0],[0,0,0,0],[t,t,t,t],[m,m,m,m],[m,m,m,m],[m,m,m,m]],
                    [[0,0,0,0],[0,0,0,0],[t,t,t,t],[m,m,m,m],[m,m,m,m],[m,m,m,m]],
                    [[0,0,0,0],[0,0,0,0],[t,t,t,t],[m,m,m,m],[m,m,m,m],[m,m,m,m]],
                    [[0,0,0,0],[0,0,0,0],[t,t,t,t],[m,m,m,m],[m,m,m,m],[m,m,m,m]],
                    
                    ])
    

    data = np.random.rand(500, 6, 4)

    

    dropout_factor = .4
    [X_train_data_formatted, X_test_data_formatted, Y_train_data_formatted, Y_test_data_formatted] = create_metadata(data)
    
    [X_train_data_formatted2, X_test_data_formatted2, Y_train_data_formatted2, Y_test_data_formatted2] = create_metadata(data)

    
    callback = callbacks.EarlyStopping(monitor='loss', patience=250, restore_best_weights=True)
    tf.keras.utils.set_random_seed(42)
    dropout_factor = 0.5
    model = Sequential()
    model.add(Dense(units=X_train_data_formatted.shape[2], input_shape=(X_train_data_formatted.shape[1], X_train_data_formatted.shape[2])))
    model.add(Dense(units=X_train_data_formatted.shape[2], activation='relu'))
    model.add(Dense(dropout_factor))
    model.add(Dense(units=X_train_data_formatted.shape[2], activation='relu'))
    model.add(Dense(dropout_factor))
    model.add(Dense(units=X_train_data_formatted.shape[2], activation='relu'))
    model.add(Dense(dropout_factor))
    model.add(Dense(units=X_train_data_formatted.shape[2], activation='relu'))
    model.add(Dense(dropout_factor))
    model.add(Dense(units=X_train_data_formatted.shape[2], activation='relu'))
    model.add(Dense(dropout_factor))
    model.add(Dense(units=X_train_data_formatted.shape[2], activation='relu'))

    model.add(Dense(units=X_train_data_formatted.shape[2]))



    model.compile(loss='mse', optimizer='adam')
    history = model.fit(x=X_train_data_formatted, y=Y_train_data_formatted, epochs=10000, validation_data=(X_test_data_formatted, Y_test_data_formatted), callbacks=[callback], shuffle=True)
    prediction = model.predict(X_test_data_formatted2)
    hp.plotLossAcc(history)

    print(X_test_data_formatted2)
    print("--------------------------------------------------------------")
    print(prediction)




def create_metadata(data):
    X = data[:, 1:]
    Y = X[:, ]
    

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.01)
    X_train_data_formatted = np.array(X_train)
    X_test_data_formatted = np.array(X_test)
    Y_train_data_formatted = np.array(Y_train)
    Y_test_data_formatted = np.array(Y_test)

    return [X_train_data_formatted, X_test_data_formatted, Y_train_data_formatted, Y_test_data_formatted]









if __name__ == "__main__":
    main()