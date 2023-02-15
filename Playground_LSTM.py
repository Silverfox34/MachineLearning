import numpy as np
from keras.models import Sequential
from sklearn.model_selection import train_test_split

def main():
    time_steps = 3
    data = np.array([[4,3,3.5,4], 
                     [1,1.25,1.3,1.2],
                     [5,4,5,3],
                     [1,2,3,2],
                     [1,5,3,4], 
                     [1,5,2,3],
                     [7,3,1,3],
                     [9,6,5,4],
                     [3,1,2,3],
                     [5, 7.312, 3, 6]])

    

    sequences = []

    for i in range(0, len(data) - time_steps):
        sequence = data[i : i+time_steps]
        sequences.append(sequence)
    
    sequences = np.array(sequences)
    print(sequences.reshape(7*4, time_steps, 1))
    

    model = Sequential()
    
    



if __name__ == "__main__":
    main()