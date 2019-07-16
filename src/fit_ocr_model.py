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


class OCRConfig(Config):
    NAME = "OCR"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 9
    STEPS_PER_EPOCH = 20
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 1024
    DETECTION_MIN_CONFIDENCE = 0.9


class DigitsDataset(utils.Dataset):
    def load_digits(self, dataset_dir: str, subset: str, annotations_path: str):
        self.add_class("digits", 1, "1")
        self.add_class("digits", 2, "2")
        self.add_class("digits", 3, "3")
        self.add_class("digits", 4, "4")
        self.add_class("digits", 5, "5")
        self.add_class("digits", 6, "6")
        self.add_class("digits", 7, "7")
        self.add_class("digits", 8, "8")
        self.add_class("digits", 9, "9")
        self.add_class("digits", 10, "0")
        # assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        annotations = json.load(open(annotations_path))
        annotations = list(annotations.values())  # don't need the dict keys
        annotations = [a for a in annotations if a['regions']]

        for a in annotations:
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes']
                            for r in a['regions'].values()]
                descriptions = [r['region_attributes']
                                for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]
                descriptions = [r['region_attributes'] for r in a['regions']]
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "digits",
                image_id=a['filename'],
                path=image_path,
                width=width, height=height,
                polygons=polygons, descriptions=descriptions)

    def load_mask(self, image_id: int) -> (np.ndarray, np.ndarray):
        image_info = self.image_info[image_id]
        if image_info["source"] != "digits":
            return super(self.__class__, self).load_mask(image_id)

        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        class_id = {}
        for i, p in enumerate(info["polygons"]):
            if p['name'] == 'polygon':
                rr, cc = skimage.draw.polygon(
                    p['all_points_y'], p['all_points_x'])
            else:
                rr, cc = skimage.draw.rectangle(
                    (p['y'], p['x']), (p['y'] + p['height'], p['x'] + p['width']))
            mask[rr, cc, i] = 1
            class_id[i] = int(info['descriptions'][i]['description'])
            if class_id[i] == 0:
                class_id[i] = 10
        class_ids = np.array([v for _, v in class_id.items()])
        return mask.astype(np.bool), class_ids.astype(np.int32)

    def image_reference(self, image_id: int):
        info = self.image_info[image_id]
        if info["source"] == "digits":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model: MaskRCNN, path_to_dataset: str = paths.IMAGES_PATH) -> None:
    # Training dataset
    dataset_train = DigitsDataset()
    dataset_train.load_digits(
        path_to_dataset, "numbers/", path_to_dataset + "numbers/ann.json")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = DigitsDataset()
    dataset_train.load_digits(
        path_to_dataset, "numbers/", path_to_dataset + "numbers/ann.json")
    dataset_val.prepare()

    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=EPOCHS_NUMBER,
                layers='heads')


def test_on_pics(model: MaskRCNN, path_to_pics: str, pics: List[str]) -> None:
    for pic in pics:
        image = skimage.io.imread(os.path.join(path_to_pics, pic))
        results = model.detect([image], verbose=1)
        r = results[0]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], ['BG', 'train_number_plates'],
                                    r['scores'])


if __name__ == "__main__":
    MODE = "train"  # eval or train
    assert MODE in ["eval", "train"]
    if MODE == "train":
        config = OCRConfig()
        model = modellib.MaskRCNN(
            mode="training", config=config, model_dir=paths.WEIGHT_LOGS_PATH)
        weights_path = 'mask_rcnn_coco.h5'
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
        # model.load_weights(weights_path, by_name=True)
        train(model, paths.IMAGES_PATH)
    else:
        class EvalConfig(OCRConfig):
            # Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        config = EvalConfig()
        model = modellib.MaskRCNN(
            mode="inference", config=config, model_dir=paths.WEIGHT_LOGS_PATH)
        weights_path = paths.WEIGHTS_PATH + "our/final_20.h5"
        model.load_weights(weights_path, by_name=True)
        test_on_pics(model, paths.IMAGES_PATH + "all_pics/",
                     ["61322186.jpg", "73372633.jpg", '52026226.jpg', '54096987.jpg'])
