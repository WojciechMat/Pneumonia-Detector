import argparse

import Detector
import prepare_input
import os

os.environ['TF_CPP_MIN_LOG_LEVER'] = '2'

arg = argparse.ArgumentParser()
arg.add_argument("--mode", help="t/r")
arg.add_argument("--input", help="path to an image")
mode = arg.parse_args().mode
img = arg.parse_args().input

train_pth = 'data/train'
test_pth = 'data/test'

n_train = 5216
n_test = 624
batch_size = 32
n_epoch = 20
n_epoch_FT = 25



