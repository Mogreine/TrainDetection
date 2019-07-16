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
from utils.all_paths import Paths
from typing import List

paths = Paths('')

EPOCHS_NUMBER = 20


class PlateConfig(Config):
    NAME = "train_number_plates"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1
    STEPS_PER_EPOCH = 398
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 1024
    DETECTION_MIN_CONFIDENCE = 0.9


def test_on_pics(model: MaskRCNN, path_to_pics: str, pics: List[str]) -> None:
    for pic in pics:
        image = skimage.io.imread(os.path.join(path_to_pics, pic))
        results = model.detect([image], verbose=1)
        r = results[0]
        # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], ['BG', 'train_number_plates'],
        #                             r['scores'])
        if len(r['rois']) == 0:
            continue 
        for i, roi in enumerate(r['rois']):
            y1, x1, y2, x2 = roi
            roi_frame = image[y1:y2, x1:x2]
            skimage.io.imsave(os.path.join('saved pics', str(i) + '_' + pic), roi_frame)
            print('save:', os.path.join('saved pics', str(i) + '_' + pic))


if __name__ == "__main__":
    class EvalConfig(PlateConfig):
        # Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = EvalConfig()
    model = modellib.MaskRCNN(
        mode="inference", config=config, model_dir=paths.WEIGHT_LOGS_PATH)
    weights_path = paths.WEIGHTS_PATH + "our/final_20.h5"
    model.load_weights(weights_path, by_name=True)
    pp = paths.IMAGES_PATH + "front_pics/"
    pics = [f for f in os.listdir(
        pp) if os.path.isfile(os.path.join(pp, f))]
    test_on_pics(model, paths.IMAGES_PATH + "front_pics/", pics)
