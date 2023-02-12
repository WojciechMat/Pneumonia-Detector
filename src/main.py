import argparse

import numpy as np
from Detector import Detector
from prepare_input import prepare_input
from keras.utils import image_dataset_from_directory
import os

os.environ['TF_CPP_MIN_LOG_LEVER'] = '2'

arg = argparse.ArgumentParser()
arg.add_argument("--mode", help="t/r")
arg.add_argument("--input", help="path to an image")
mode = arg.parse_args().mode
img = arg.parse_args().input

train_pth = '../data/train'
test_pth = '../data/test'

batch_size = 32
n_epoch = 20
n_epoch_FT = 25

train_gen = image_dataset_from_directory(
                train_pth,
                image_size=(700, 700),
                color_mode='grayscale',
                label_mode='categorical',
                batch_size=batch_size,
                smart_resize=1./255
            )
test_gen = image_dataset_from_directory(
                test_pth,
                image_size=(700, 700),
                color_mode='grayscale',
                label_mode='categorical',
                batch_size=batch_size,
                smart_resize=1./255
            )
detector = Detector()
detector.summary()
if mode == 't':
    detector.train(lr=0.001, train_db=train_gen, test_db=test_gen, epochs=n_epoch)
    detector.prepare_for_FT()
    detector.train(lr=0.0001, train_db=train_gen, test_db=test_gen, epochs=n_epoch_FT)
    detector.save_weights('detection.h5')
elif mode == 'r':
    detector.load_weights('detection.h5')
    image = prepare_input(img)
    prediction = detector.predict(image)
    print(np.argmax(prediction))



