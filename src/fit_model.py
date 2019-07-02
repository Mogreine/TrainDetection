import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

ROOT_DIR = '../'

sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize

COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "logs/mask_rcnn_coco.h5")
NOMEROFF_NET_WEIGHTS_PATH = os.path.join(ROOT_DIR, "logs/nomeroff_net.h5")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs/")



class PlateConfig(Config):
    NAME = "train_number_plates"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1
    STEPS_PER_EPOCH = 100
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 1024
    DETECTION_MIN_CONFIDENCE = 0.7


class PlateDataset(utils.Dataset):

    def load_plates(self, dataset_dir, subset):
        self.add_class("train_number_plates", 1, "train_number_plates")
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        annotations = json.load(open("../data/via/via_export_json.json"))
        annotations = list(annotations.values())  # don't need the dict keys
        annotations = [a for a in annotations if a['regions']]

        for a in annotations:
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "train_number_plates",
                image_id=a['filename'],
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        if image_info["source"] != "train_number_plates":
            return super(self.__class__, self).load_mask(image_id)

        info = self.image_info[image_id]
        # print(info)
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            #print("load_MASK DATA:")
            #print(i, p)
            rr, cc = skimage.draw.rectangle((p['y'], p['x']), (p['y'] + p['height'], p['x'] + p['width']))
            mask[rr, cc, i] = 1
            # print(mask[rr, cc, i])
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "train_number_plates":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model, path_to_dataset):
    # Training dataset
    dataset_train = PlateDataset()
    dataset_train.load_plates(path_to_dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = PlateDataset()
    dataset_val.load_plates(path_to_dataset, "val")
    dataset_val.prepare()

    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')
    

def test_on_pics(model, path_to_pics, pics):
    for pic in pics:
        image = skimage.io.imread(os.path.join(path_to_pics, pic))
        results = model.detect([image], verbose = 1)
        r = results[0]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], ['BG', 'train_number_plates'], r['scores'])


if __name__ == "__main__":
    MODE = "eval"  # eval or train
    assert MODE in ["eval", "train"]
    if MODE == "train":
        config = PlateConfig()
        model = modellib.MaskRCNN(mode="training", config=config, model_dir=DEFAULT_LOGS_DIR)
        weights_path = COCO_WEIGHTS_PATH
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
        # model.load_weights(weights_path, by_name=True)
        train(model, "../data/images")
    else:
        class EvalConfig(PlateConfig):
            # Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1


        config = EvalConfig()
        model = modellib.MaskRCNN(mode="inference", config=config, model_dir=DEFAULT_LOGS_DIR)
        weights_path = "../logs/weights/mask_rcnn_train_number_plates_0007.h5"
        model.load_weights(weights_path, by_name=True)
        test_on_pics(model, "../data/images/new_pics", ["5.jpg"])