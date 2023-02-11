from keras.layers import Conv2D, AveragePooling2D, BatchNormalization,\
    Dense, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import Adam

class Detector:
    __model = Sequential()

    def __init__(self):
        model = Sequential()
        model.add(Conv2D(16, kernel_size=(3, 3), input_shape=(700, 700, 1), activation='relu'))

        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        model.add(Dropout(0.3))

        model.add(AveragePooling2D(2, 2))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(Dropout(0.3))

        model.add(AveragePooling2D(21, 21))
        model.add(BatchNormalization())

        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(2, activation='softmax'))

        model.layers[11].trainable = False
        model.layers[12].trainable = False
        model.layers[13].trainable = False

        self.__model = model

    def model(self) -> Sequential:
        return self.__model

    def prepare_for_FT(self):
        for layer in self.__model.layers:
            if layer.trainable:
                layer.trainable = False
            else:
                layer.trainable = True

    def load_weights(self, model_path):
        self.__model.load_weights(model_path)

    def train(self, lr):


