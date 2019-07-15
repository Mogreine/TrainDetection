import numpy as np
from mrcnn.config import Config
from mrcnn.model import MaskRCNN, utils
from mrcnn import visualize
import src.fit_model as fit_model
import cv2
from src.utils.all_paths import Paths
from src.fit_model import PlateConfig, PlateDataset
from typing import List, Tuple

paths = Paths('../')


def add_mask(img: np.ndarray, masks: List, color: Tuple[int, int, int]) -> np.ndarray:
    for mask in masks:
        img = visualize.apply_mask(img, mask, color)
    return img


def add_boxes(img: np.ndarray, rois) -> np.ndarray:
    for roi in rois:
        y1, x1, y2, x2 = roi
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2, 4)
    return img


def add_instances(image: np.ndarray, boxes: np.array, masks: List,
                  class_ids: np.ndarray, class_names: List[str], scores: np.ndarray = None) -> np.ndarray:
    N = boxes.shape[0]
    if not N:
        return image
    colors = visualize.random_colors(N)

    for i in range(N):
        color = colors[i]
        if not np.any(boxes[i]):
            continue
        y1, x1, y2, x2 = boxes[i]
        cv2.rectangle(image, (x1, y1), (x2, y2), (color[0], color[1], color[2]), 1, 1)

    return image


# start time and duration in seconds
def video_detect(path_to_video: str, path_to_save: str, model: MaskRCNN,
                 start_time: int = 0, duration: int = 10000000) -> None:
    capture = cv2.VideoCapture(path_to_video)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))

    vwriter = cv2.VideoWriter(path_to_save,
                              cv2.VideoWriter_fourcc(*'MJPG'),
                              fps, (width, height))
    count = 0
    while capture.isOpened() and count < fps * (start_time + duration):
        success, frame = capture.read()
        print(f"frame: {count}")
        if not success:
            break
        if count < start_time * fps:
            count += 1
            continue
        # frame = frame[..., ::-1]
        r = model.detect([frame], verbose=0)[0]
        frame = add_instances(frame, r['rois'], r['masks'], r['class_ids'], ['BG', 'train_number_plates'], r['scores'])
        # frame = frame[..., ::-1]
        vwriter.write(frame)
        count += 1
    vwriter.release()
    print("Saved to ", path_to_save)


if __name__ == "__main__":
    class EvalConfig(PlateConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1


    config = EvalConfig()
    model = MaskRCNN(mode="inference", config=config, model_dir=paths.WEIGHT_LOGS_PATH)
    weights_path = paths.WEIGHTS_PATH + "our/final_20.h5"
    model.load_weights(weights_path, by_name=True)
    video_detect(paths.VIDEOS_PATH + 'angle.mp4', paths.VIDEOS_PATH + 'predict.mp4', model)
