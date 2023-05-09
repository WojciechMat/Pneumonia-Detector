import cv2
import numpy as np


def prepare_input(img_path):
    img = np.expand_dims(
        np.expand_dims(
            cv2.resize(
                cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY),
                (300, 300)
            ),
            -1
        ),
        0
    )


