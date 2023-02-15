import numpy as np
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers import LSTM as LSTM

def main():
    time_steps = 3
    data = np.array([
        
                    [[1,2,3, 2,3,4, 3,4,5, 4,5,6],

                    [10,11,12,11,12,13,12,13,14,13,14,15]],
                    
                    [[1,2,3, 2,3,4, 3,4,5, 4,5,6],

                    [10,11,12,11,12,13,12,13,14,13,14,15]],

                    [[1,2,3, 2,3,4, 3,4,5, 4,5,6],

                    [10,11,12,11,12,13,12,13,14,13,14,15]]
                    
                    ])
    print(data.shape)

    return

    #sequences = []

    #for i in range(0, len(data) - time_steps):
        #sequence = data[i : i+time_steps]
        #sequences.append(sequence)
    
    #sequences = np.array(sequences)
    #print(sequences.reshape(7, time_steps*4, 1))
    print()

    model = Sequential()
    model.add(LSTM(units=32, input_shape=(data.shape[1], 1)))
    
    model.compile(loss='mse', optimizer='adam')
    model.fit(data, data)


if __name__ == "__main__":
    main()