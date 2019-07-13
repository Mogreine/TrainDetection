import numpy as np
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize
import src.fit_model as fit_model
import cv2
from src.utils.all_paths import Paths
from src.fit_model import PlateConfig, PlateDataset

paths = Paths('../')


def add_mask(img, masks, color):
    for mask in masks:
        img = visualize.apply_mask(img, mask, color)
    # if mask.shape[-1] > 0:
    #     mask = (np.sum(mask, -1, keepdims=True) >= 1)
    #     img = np.where(mask, img).astype(np.uint8)
    return img

def add_boxes(img, rois):
    for roi in rois:
        y1, x1, y2, x2 = roi
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2, 4)
    return img


def add_instances(image, boxes, masks, class_ids, class_names, scores = None):
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

def video_detect(path_to_video, path_to_save, model):
    capture = cv2.VideoCapture(path_to_video)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))

    vwriter = cv2.VideoWriter(path_to_save,
                            cv2.VideoWriter_fourcc(*'MJPG'),
                            fps, (width, height))
    count = 0
    while capture.isOpened():
        success, frame = capture.read()
        print(f"frame: {count}")
        if not success:
            break
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
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=paths.WEIGHT_LOGS_PATH)
    weights_path = paths.WEIGHTS_PATH + "our/final_20.h5"
    model.load_weights(weights_path, by_name=True)
    video_detect(paths.VIDEOS_PATH + 'test.mp4', paths.VIDEOS_PATH + 'predict.mp4', model)