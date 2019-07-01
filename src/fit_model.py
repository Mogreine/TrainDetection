import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

PATH_TO_IMG = "data/images/tmp"


class DSConfig(Config):
    NAME = "wheel"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8
    NUM_CLASSES = 1 + 1

    IMAGE_MIN_DIM = 128 # fix
    IMAGE_MAX_DIM = 128 # fix

    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    TRAIN_ROIS_PER_IMAGE = 32

    STEPS_PER_EPOCH = 100

    VALIDATION_STEPS = 5

class NewDataset(utils.Dataset):
    def load_wheels(self):
        self.add_class("wheels", 1, "train_wheel")



    def load_image(self, image_id):
        pass
    
    def load_mask(self, image_id):
        pass
    
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        pass

# config = DSConfig()

# test = NewDataset()
# file = open("data/via/via_export_coco.json", "r")
# print(file.read())
# file.close()
coco = COCO("data/via/via_export_coco.json")
test = coco.getCatIds()
print(test)