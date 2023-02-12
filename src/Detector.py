from keras.layers import Conv2D, AveragePooling2D, BatchNormalization,\
    Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import image_dataset_from_directory


class Detector:
    __model = Sequential()

    def __init__(self):
        self.__model.add(Conv2D(16, kernel_size=(3, 3), input_shape=(700, 700, 1), activation='relu'))

        self.__model.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))
        self.__model.add(Dropout(0.4))

        self.__model.add(AveragePooling2D(7, 7))
        self.__model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        self.__model.add(Dropout(0.4))

        self.__model.add(BatchNormalization())

        self.__model.add(AveragePooling2D(7, 7))
        self.__model.add(Dropout(0.3))
        self.__model.add(Conv2D(32, kernel_size=(3, 3)))

        self.__model.add(MaxPooling2D(4, 4))

        self.__model.add(Flatten())
        self.__model.add(Dense(32, activation='relu'))
        self.__model.add(Dropout(0.2))
        self.__model.add(Dense(32, activation='relu'))
        self.__model.add(Dropout(0.5))
        self.__model.add(Dense(16, activation='relu'))
        self.__model.add(Dense(2, activation='softmax'))

        self.__model.layers[14].trainable = False
        self.__model.layers[15].trainable = False
        self.__model.layers[16].trainable = False

    def model(self) -> Sequential():
        return self.__model

    def summary(self):
        self.__model.summary()

    def predict(self, img):
        return self.__model.predict(img)

    def prepare_for_FT(self):
        for layer in self.__model.layers:
            if layer.trainable:
                layer.trainable = False
            else:
                layer.trainable = True

    def load_weights(self, model_path):
        self.__model.load_weights(model_path)

    def train(self, lr, train_db, test_db, epochs):
            n_train = 5216
            n_test = 624
            batch_size = 32
            early_stop = EarlyStopping(monitor='val_loss', restore_best_weights=True, patience=5, verbose=1)
            callback = [early_stop]
            self.__model.compile(loss='categorical_crossentropy',
                                 optimizer=Adam(learning_rate=lr, decay=1e-6),
                                 metrics=['accuracy'])
            self.__model.info = self.__model.fit(
                train_db,
                steps_per_epoch=n_train // batch_size,
                epochs=epochs,
                validation_data=test_db,
                validation_steps=n_test // batch_size,
                callbacks=callback,
                use_multiprocessing=True
            )

    def save_weights(self, path):
        self.__model.save_weights(path)




