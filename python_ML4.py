import tensorflow as tf
from ImageDataLoader import ImageGenerator as IG
import sys
import numpy as np


def main():
    np.set_printoptions(threshold=sys.maxsize)
    #SOURCE: ROCK PAPER SCISSORS
    #https://laurencemoroney.com/datasets.html
    ig = IG()
    WIDTH = 300
    HEIGHT = 300

    train_data = ig.LoadImageDataFromFile("C:\Users\Moritz\Desktop\Allgemeines\MachineLearning\rps", WIDTH, HEIGHT)
    val_data = ig.LoadImageDataFromFile(":\Users\Moritz\Desktop\Allgemeines\MachineLearning\rps", WIDTH, HEIGHT)


    TRAINING_DIR = "C:/Users/LS_MFE/Desktop/rps"
    VALIDATION_DIR = "C:/Users/LS_MFE/Desktop/rps-test-set"
    training_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)
    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)

    train_generator = training_datagen.flow_from_directory(TRAINING_DIR, target_size=(150,150), class_mode='categorical')
    validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR, target_size=(150,150), class_mode='categorical')
    
    validataion_dataset_paper = validation_datagen.flow_from_directory("YOUR PATH TO PAPER ONLY DATA, FOR TESTING PURPOSES", target_size=(150,150), class_mode='categorical')
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (3,3) , activation='relu', input_shape=(WIDTH, HEIGHT,3)))
    model.add(tf.keras.layers.MaxPooling2D(2,2))
    model.add(tf.keras.layers.Conv2D(64, (3,3) , activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(2,2))
    model.add(tf.keras.layers.Conv2D(128, (3,3) , activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(2,2))
    model.add(tf.keras.layers.Conv2D(128, (3,3) , activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(2,2))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))



    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics='accuracy')

    history = model.fit_generator(train_data, epochs=3, validation_data=val_data, verbose=1)
    print[history]

    #Should only detect rock here
    #classes = model.predict(val_data)
    
    

    #print(classes)








if __name__ == "__main__":
    main()
