import numpy as np
from mrcnn.config import Config
from mrcnn.model import MaskRCNN, utils
from mrcnn import visualize
import fit_model as fit_model
import fit_ocr_model as fit_ocr_model
import cv2
from src.utils.all_paths import Paths
from src.fit_model import PlateConfig
from src.fit_ocr_model import OCRConfig
from src.utils.recognizer import RecognizeHelper
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
                  class_ids: np.ndarray, class_names: List[str], detected_numbers: List, scores: np.ndarray = None) -> np.ndarray:
    N = boxes.shape[0]
    if not N:
        return image
    colors = visualize.random_colors(N)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(N):
        color = colors[i]
        if not np.any(boxes[i]):
            continue
        y1, x1, y2, x2 = boxes[i]
        image = cv2.rectangle(image, (x1, y1), (x2, y2),
                              (color[0], color[1], color[2]), 1, 1)
        image = cv2.putText(image, str(detected_numbers[i]),
                            ((x1 + x2) // 2, y1), font, 1, (255, 0, 0), 1, cv2.LINE_AA)
    return image


# start time and duration in seconds
def video_detect(path_to_video: str, path_to_save: str, detect_model: MaskRCNN, ocr_model: MaskRCNN,
                 start_time: int = 0, duration: int = 10000000) -> None:
    rec_helper = RecognizeHelper()
    capture = cv2.VideoCapture(path_to_video)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    all_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    vwriter = cv2.VideoWriter(path_to_save,
                              cv2.VideoWriter_fourcc(*'MJPG'),
                              fps, (width, height))
    count = 0
    all_time = 0
    while capture.isOpened() and count < fps * (start_time + duration):
        e1 = cv2.getTickCount()
        success, frame = capture.read()
        # print(f"frame: {count}")
        if not success:
            break
        if count < start_time * fps:
            count += 1
            continue
        res_detect = detect_model.detect([frame], verbose=0)[0]
        detected_numbers = []
        for roi in res_detect['rois']:
            y1, x1, y2, x2 = roi
            number_plate = frame[y1:y2, x1:x2]
            r = ocr_model.detect([number_plate], verbose=0)[0]
            detected_number: str = rec_helper.get_number(
                r['rois'], r['class_ids'])
            detected_numbers.append(detected_number)
        frame = add_instances(frame, res_detect['rois'], res_detect['masks'], res_detect['class_ids'], [
                              'BG', 'train_number_plates'], detected_numbers, res_detect['scores'])

        vwriter.write(frame)
        count += 1
        e2 = cv2.getTickCount()
        time = (e2 - e1) / cv2.getTickFrequency()
        all_time += time
        eta = (all_frame_count - count) * all_time / count
        print(
            f'on frame {count} with ETA: {eta} sec or {eta // 60} min')
    vwriter.release()
    print("Saved to ", path_to_save)


if __name__ == "__main__":
    class EvalDetectConfig(PlateConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    class EvalOCRConfig(OCRConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = EvalDetectConfig()
    detect_model = MaskRCNN(mode="inference", config=config,
                            model_dir=paths.WEIGHT_LOGS_PATH)
    weights_path = paths.WEIGHTS_PATH + "our/final_20.h5"
    detect_model.load_weights(weights_path, by_name=True)

    config = EvalOCRConfig()
    ocr_model = MaskRCNN(mode="inference", config=config,
                         model_dir=paths.WEIGHT_LOGS_PATH)
    weights_path = paths.WEIGHTS_PATH + "our/rough_aug_60.h5"
    ocr_model.load_weights(weights_path, by_name=True)

    video_detect(paths.VIDEOS_PATH + 'vid1_new.mp4',
                 paths.VIDEOS_PATH + 'predict.mp4', detect_model, ocr_model)
