import os
import json
import numpy as np
import skimage.draw
from mrcnn import utils
from src.utils.all_paths import Paths
from typing import List, Dict

paths = Paths('../../')


class GeneralDataset(utils.Dataset):
    def __init__(self, dataset_name: str, dataset_dir: str, annotations_name: str, class_map=None):
        super().__init__(class_map=None)
        self.dataset_name_ = dataset_name
        self.dataset_dir_ = dataset_dir
        self.set_size = 0
        self.annotations_name = annotations_name

    def load(self, annotations_path: str = None):
        if annotations_path is None:
            annotations_path = self.dataset_dir_
        annotations_path += self.annotations_name
        annotations = json.load(open(annotations_path))
        annotations = list(annotations.values())  # don't need the dict keys
        annotations = [a for a in annotations if a['regions']]
        self.set_size = len(annotations)
        self.load_from_annotations(annotations)

    def load_from_annotations(self, annotations: List[Dict]):
        for a in annotations:
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
                descriptions = [r['region_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]
                descriptions = [r['region_attributes'] for r in a['regions']]
            image_path = os.path.join(self.dataset_dir_, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(self.dataset_name_,
                           image_id=a['filename'],
                           path=image_path,
                           width=width, height=height,
                           polygons=polygons, descriptions=descriptions)

    def load_mask(self, image_id: int) -> (np.ndarray, np.ndarray):
        image_info = self.image_info[image_id]
        if image_info["source"] != self.dataset_name_:
            return super(self.__class__, self).load_mask(image_id)

        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        class_id = {}
        for i, p in enumerate(info["polygons"]):
            if p['name'] == 'polygon':
                rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            else:
                rr, cc = skimage.draw.rectangle((p['y'], p['x']), (p['y'] + p['height'], p['x'] + p['width']))
            mask[rr, cc, i] = 1
            class_id[i] = int(info['descriptions'][i]['description'])
            if class_id[i] == 0:
                class_id[i] = 10
        class_ids = np.array([v for _, v in class_id.items()])
        return mask.astype(np.bool), class_ids.astype(np.int32)

    def image_reference(self, image_id: int):
        info = self.image_info[image_id]
        if info["source"] == self.dataset_name_:
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


class PlateDataset(GeneralDataset):
    def __init__(self, dataset_dir, annotations_name: str):
        super().__init__('train_number_plates', dataset_dir, annotations_name)
        self.add_class("train_number_plates", 1, "train_number_plates")


class DigitDataset(GeneralDataset):
    def __init__(self, dataset_dir, annotations_name: str):
        super().__init__('digits', dataset_dir, annotations_name)
        for i in range(1, 11):
            self.add_class('digits', i, str(i % 10))
