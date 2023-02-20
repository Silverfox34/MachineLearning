from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

class ImageGenerator:
    def __init__(self):
        pass

    def LoadImageDataFromFile(self, PATH, width : int, height : int):
        """ Simplifies the image data loading process by taking only small amount
            of information about the folder structure to provide the user
            with the images needed. The output of this method can directly
            be put in model.fit(...) as an argument.

        Args:
            PATH (String): The path to the folder where the class folders lay
            width (int): width of the pictures, has to be the same for every picture
            height (int): height of the pictures, has to be the same for every picture

        Returns:
            Generated images from the source file PATHs folder sructure
        """
        ImageDatagenerator = keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)
        generatedImages = ImageDatagenerator.flow_from_directory(PATH, target_size=(width, height), class_mode='categorical')

        return generatedImages

class HistoryPlotter:
    def __init__(self):
        pass

    def plotLossAcc(history):
        plt.rcParams["figure.figsize"] = (13,4)

        plt.subplot(1, 2, 1)
        plt.plot(history.history['categorical_accuracy'], label='Accuracy', color="tab:blue")
        plt.plot(history.history['loss'], label='Loss', color="tab:orange")

        plt.title('Training')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(visible=True)


        plt.subplot(1, 2, 2)
        plt.plot(history.history['val_categorical_accuracy'], label='Validation Accuracy', color="tab:blue")
        plt.plot(history.history['val_loss'], label='Validation Loss', color="tab:orange")

        plt.title('Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Accuracy')
        plt.legend()
        plt.grid(visible=True)

        plt.tight_layout()
        plt.show()

        del history   

