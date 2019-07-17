import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
import skimage.draw
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn.model import MaskRCNN
from mrcnn import visualize
from src.utils.all_paths import Paths
from src.datasets.Datasets import DigitDataset
from src.models.OCR import OCR
from typing import List

paths = Paths('../')

EPOCHS_NUMBER = 60
STEPS = 120


if __name__ == "__main__":
    MODE = "eval"  # eval or train
    assert MODE in ["eval", "train"]
    if MODE == "train":
        train_ds = DigitDataset(dataset_dir=paths.IMAGES_PATH + 'numbers_aug', annotations_name='all.json')
        val_ds = DigitDataset(dataset_dir=paths.IMAGES_PATH + 'numbers', annotations_name='fixed_ones.json')
        ocr = OCR(image_min_dim=16, image_max_dim=512)
        ocr.fit(train_ds, val_ds, epochs=30)
    else:
        numbers_path = paths.IMAGES_PATH + 'numbers/'
        files = os.listdir(numbers_path)
        images = [skimage.io.imread(numbers_path + file_name)
                  for file_name in files if file_name[len(file_name) - 3:] == 'jpg']
        ocr = OCR()

        true_numbers = [file[2:-4] for file in files]
        for i in range(len(true_numbers)):
            true_numbers[i] = ''.join([j for j in true_numbers[i] if 48 <= ord(j) <= 57])
        score = ocr.score(images, true_numbers)
        print(score)
