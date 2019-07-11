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
import matplotlib.pyplot as plt
from src.utils.all_paths import Paths

paths = Paths()


class Evaluator(object):
    def __init__(self, model, path_to_dataset, path_to_ann):
        self.model = model
        self.dataset_val = fit_model.PlateDataset()
        self.dataset_val.load_plates(path_to_dataset, "side_pics/init/val/", path_to_ann)
        self.dataset_val.prepare()

    def eval(self, config, threshold=0.5):
        image_ids = self.dataset_val.image_ids
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
                                 r["rois"], r["class_ids"], r["scores"], r['masks'], iou_threshold=threshold)
            APs.append(AP)
        # print("mAP: ", np.mean(APs))
        return np.mean(APs)


if __name__ == "__main__":
    class EvalConfig(fit_model.PlateConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1


    config = EvalConfig()
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=paths.WEIGHT_LOGS_PATH)
    model.load_weights(paths.WEIGHTS_PATH + 'our/side_20.h5', by_name=True)

    eval = Evaluator(model, paths.IMAGES_PATH, paths.ANNOTATIONS_PATH + 'new/test_plates_polygon.json')
    aps = []
    x = []
    for i in range(50, 100, 5):
        ap = eval.eval(config, i / 100)
        aps.append(ap)
        x.append(i / 100)
    print("mAP: {:.3f}", np.mean(aps))
    plt.plot(x, aps)
    plt.ylabel('AP')
    plt.xlabel('Recall threshold')
    plt.suptitle('mAP over threshold')
    plt.show()

