import numpy as np


class RecognizeHelper(object):
    def get_number(self, rois: np.ndarray, class_ids: np.array) -> str:
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
