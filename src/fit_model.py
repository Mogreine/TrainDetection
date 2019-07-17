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
from typing import List
from src.datasets.Datasets import PlateDataset
from src.models.OCR import OCR

paths = Paths('../')

EPOCHS_NUMBER = 20


class PlateConfig(Config):
    NAME = "train_number_plates"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1
    STEPS_PER_EPOCH = 398
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 1024
    DETECTION_MIN_CONFIDENCE = 0.9

if __name__ == "__main__":
    MODE = "eval"  # eval or train
    assert MODE in ["eval", "train"]
    if MODE == "train":
        config = PlateConfig()
        model = modellib.MaskRCNN(mode="training", config=config, model_dir=paths.WEIGHT_LOGS_PATH)
        weights_path = paths.NOMEROFF_NET_WEIGHTS_PATH
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
        # model.load_weights(weights_path, by_name=True)
        train(model, paths.IMAGES_PATH)
    else:
        class EvalConfig(PlateConfig):
            # Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        config = EvalConfig()
        model = modellib.MaskRCNN(mode="inference", config=config, model_dir=paths.WEIGHT_LOGS_PATH)
        weights_path = paths.WEIGHTS_PATH + "our/final_20.h5"
        model.load_weights(weights_path, by_name=True)
        test_on_pics(model, paths.IMAGES_PATH + "all_pics/",
                     ["61322186.jpg", "73372633.jpg", '52026226.jpg', '54096987.jpg'])

if __name__ == "__main__":
    MODE = "eval"  # eval or train
    assert MODE in ["eval", "train"]
    if MODE == "train":
        train_ds = PlateDataset(dataset_dir=paths.IMAGES_PATH + 'all_pics_aug', annotations_name='ann.json')
        val_ds = PlateDataset(dataset_dir=paths.IMAGES_PATH + 'all_pics_aug', annotations_name='hundred.json')
        ocr = OCR(image_min_dim=128, image_max_dim=1024)
        ocr.fit(train_ds, val_ds, epochs=20)
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
