from mrcnn import model as modellib
from mrcnn.model import MaskRCNN
from mrcnn import visualize
from src.utils.all_paths import Paths
from src.datasets.Datasets import DigitDataset
from src.models.ModelConfig import ModelConfig
import numpy as np
from typing import List, Union, Tuple
import os

paths = Paths('../')


class OCR:
    def __init__(self, name: str = 'OCR', num_classes: int = 10, image_min_dim: int = 16, image_max_dim: int = 512):
        self.config = ModelConfig(name=name,
                                  num_classes=num_classes,
                                  image_min_dim=image_min_dim,
                                  image_max_dim=image_max_dim,
                                  steps=120)
        self.config.IMAGE_META_SIZE = 23

    def fit(self, train_dataset: DigitDataset, val_dataset: DigitDataset,
            epochs: int = 20, layers: str = 'heads', logs_path: str = paths.WEIGHT_LOGS_PATH,
            weights_path: str = paths.WEIGHTS_PATH + 'pretrained/mask_rcnn_coco.h5') -> None:
        train_dataset.load()
        train_dataset.prepare()
        val_dataset.load()
        val_dataset.prepare()

        self.config.EPOCHS = epochs
        self.config.STEPS_PER_EPOCH = train_dataset.set_size

        model = MaskRCNN(mode='training', config=self.config, model_dir=logs_path)
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])

        print("Training network")
        model.train(train_dataset, val_dataset,
                    learning_rate=self.config.LEARNING_RATE,
                    epochs=self.config.EPOCHS,
                    layers=layers)

    def predict(self, pictures: Union[DigitDataset, List[np.ndarray]],
                weights_path: str = paths.WEIGHTS_PATH + 'our/rough_aug_60.h5') -> List[str]:
        model = modellib.MaskRCNN(mode="inference", config=self.config, model_dir=paths.WEIGHT_LOGS_PATH)
        model.load_weights(weights_path, by_name=True)


        preds = []
        for pic in pictures:
            image = pic
            results = model.detect([image], verbose=0)
            r = results[0]
            ids = ['BG'] + [str(i) for i in range(1, 11)]
            visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], ids,
                                        r['scores'])

            number = self.get_number(r['rois'], r['class_ids'])
            preds.append(number)
        return preds

    def score(self, x: Union[DigitDataset, List[np.ndarray]], y_true: List[str],
              quotient: bool = True) -> Union[float, Tuple[int, int]]:
        y_predicted = self.predict(x)
        correct = 0
        for i in range(len(y_true)):
            if y_true[i] == y_predicted[i]:
                correct += 1
        if quotient:
            return correct, len(y_true)
        else:
            return correct / len(y_true)

    def get_number(self, rois: np.array, class_ids: np.array) -> str:
        centroids = []
        for i in range(len(rois)):
            y1, x1, y2, x2 = rois[i]
            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
            class_id = class_ids[i]
            if class_id == 10:
                class_id = 0
            centroids.append((centroid, str(class_id)))
        centroids.sort()
        number = ''.join([s[1] for s in centroids])
        return number
