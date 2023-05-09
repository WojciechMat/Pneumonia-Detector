from keras.layers import Conv2D, AveragePooling2D, BatchNormalization,\
    Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import image_dataset_from_directory


class Detector:

    def __init__(self):
        self.__model = Sequential([
            Conv2D(32, kernel_size=(3, 3), input_shape=(300, 300, 1), activation='relu'),
            Conv2D(32, kernel_size=(3, 3), activation="relu"),
            AveragePooling2D(5, 5),
            Conv2D(64, kernel_size=(3, 3), activation="relu"),
            AveragePooling2D(5, 5),
            Conv2D(64, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Flatten(),

            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(16, activation='relu'),
            Dense(8, activation='relu'),
            Dense(2, activation='softmax')
        ])


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




