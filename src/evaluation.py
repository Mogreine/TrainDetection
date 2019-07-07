import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize
import src.fit_model as fit_model

ROOT_DIR = "../"
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs/")
WEIGHTS_PATH = "../logs/with_aug.h5"


class Evaluator(object):
    def __init__(self, model, path_to_dataset, path_to_ann):
        self.model = model
        self.dataset_val = fit_model.PlateDataset()
        self.dataset_val.load_plates(path_to_dataset, "val", path_to_ann)
        self.dataset_val.prepare()

    def eval(self, config):
        image_ids = np.random.choice(self.dataset_val.image_ids, 10)
        APs = []
        for image_id in image_ids:
            image, image_meta, gt_class_id, gt_bbox, gt_mask = \
                modellib.load_image_gt(self.dataset_val, config,
                                       image_id, use_mini_mask=False)
            molded_images = np.expand_dims(modellib.mold_image(image, config), 0)
            results = model.detect([image], verbose=0)
            r = results[0]
            AP, precisions, recalls, overlaps = \
                utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                 r["rois"], r["class_ids"], r["scores"], r['masks'])
            APs.append(AP)
        print("mAP: ", np.mean(APs))


if __name__ == "__main__":
    class EvalConfig(fit_model.PlateConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1


    config = EvalConfig()
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=DEFAULT_LOGS_DIR)
    model.load_weights(WEIGHTS_PATH, by_name=True)

    eval = Evaluator(model, '../data/images/all_pics/', '../data/via/ann.h5')
    eval.eval(config)
