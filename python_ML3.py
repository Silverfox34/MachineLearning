import numpy as np
import pandas 
import sys
from tensorflow import keras
from keras.models import Sequential



def main():
    np.set_printoptions(threshold=sys.maxsize)
    
    data = pandas.read_csv('Seed_Data.csv')
    data = data[data.notna()]
    #data = normalize(data)


    train_dataset = data.sample(frac=0.98)
    test_dataset = data.drop(train_dataset.index)
    train_labels = train_dataset.pop('A')
    test_labels = test_dataset.pop('A')




    model = Sequential()
    model.add(keras.layers.Dense(units = 7, activation = 'relu', input_dim = 7))
    model.add(keras.layers.Dense(units = 4, activation = 'relu'))
    model.add(keras.layers.Dense(units = 1))
    
    

    model.compile(loss = 'mse', optimizer='rmsprop')


    
    history = model.fit(train_dataset, train_labels, epochs = 500)

    test_pred = model.predict(test_dataset)
    print(test_labels)
    print(test_pred)
    

def normalize(dataset):
    return ((dataset - dataset.mean()) / dataset.std())



if __name__ == "__main__":
    main()