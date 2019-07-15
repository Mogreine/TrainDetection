import os
import time
import cv2
import numpy as np
from matplotlib import pyplot as plt
from src.utils.all_paths import Paths

paths = Paths('../')


class Rectangle(object):
    def __init__(self, x, y, width=None, height=None, x2=None, y2=None):
        assert width is not None and height is not None or x2 is not None and y2 is not None
        self.lu_x = x
        self.lu_y = y
        if width is None:
            self.width = x2 - x
            self.height = y2 - y
            self.rd_x = x2
            self.rd_y = y2
        else:
            self.width = width
            self.height = height
            self.rd_x = x + width
            self.rd_y = y + height

    def get_points(self):
        return self.lu_x, self.lu_y, self.rd_x, self.rd_y

    def get_with_params(self):
        return self.lu_x, self.lu_y, self.width, self.height


class TrainCounter(object):
    train_count = 0
    threshold = 20

    def __init__(self, path_to_video: str):
        self.path_to_video = path_to_video

    def camshift(self, rect: Rectangle):
        capture = cv2.VideoCapture(self.path_to_video)
        ret, frame = capture.read()
        track_window = rect.get_with_params()
        roi = frame[rect.lu_y:rect.lu_y + rect.height,
              rect.lu_x:rect.lu_x + rect.width]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)),
                           np.array((180., 255., 255.)))
        roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        rect.lu_x += 200
        while capture.isOpened():
            ret, frame = capture.read()
            if not ret:
                break
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            ret, track_window = cv2.CamShift(dst, track_window, term_crit)
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            mm = pts.min(axis=0)[0]
            if mm <= self.threshold:
                self.train_count += 1
                print(self.train_count)
                track_window = rect.get_with_params()
                roi = frame[rect.lu_y:rect.lu_y + rect.height,
                      rect.lu_x:rect.lu_x + rect.width]
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv_roi, np.array(
                    (0., 60., 32.)), np.array((180., 255., 255.)))
                roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
                cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
            img2 = cv2.polylines(frame, [pts], True, 255, 2)
            cv2.imshow('img2', img2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        capture.release()
        cv2.destroyAllWindows()

    def meanshift(self, rect: Rectangle, path_to_save: str = None):
        capture = cv2.VideoCapture(self.path_to_video)
        if path_to_save is not None:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(capture.get(cv2.CAP_PROP_FPS))
            vwriter = cv2.VideoWriter(path_to_save,
                                      cv2.VideoWriter_fourcc(*'MJPG'),
                                      fps, (width, height))
        ret, frame = capture.read()
        track_window = rect.get_with_params()
        roi = frame[rect.lu_y:rect.lu_y + rect.height,
              rect.lu_x:rect.lu_x + rect.width]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)  # COLOR_BGR2HSV
        mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)),
                           np.array((180., 255., 255.)))
        roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(roi_hist, roi_hist, 0, 255,
                      cv2.NORM_MINMAX)  # NORM_MINMAX
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        rect.lu_x += 100
        font = cv2.FONT_HERSHEY_SIMPLEX
        while capture.isOpened():
            ret, frame = capture.read()
            if not ret:
                break
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # COLOR_BGR2HSV
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            ret, track_window = cv2.meanShift(dst, track_window, term_crit)
            x, y, w, h = track_window
            mm = x
            if mm <= self.threshold:
                self.train_count += 1
                print(self.train_count)
                track_window = rect.get_with_params()
                roi = frame[rect.lu_y:rect.lu_y + rect.height,
                      rect.lu_x:rect.lu_x + rect.width]
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv_roi, np.array(
                    (0., 60., 32.)), np.array((180., 255., 255.)))
                roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
                cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
            img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
            if path_to_save is not None:
                img2 = cv2.putText(img2, str(self.train_count),
                                   (10, 500), font, 4, (255, 0, 0), 2, cv2.LINE_AA)
                vwriter.write(img2)
            cv2.imshow('img2', img2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        capture.release()
        if path_to_save is not None:
            vwriter.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tc = TrainCounter(paths.VIDEOS_PATH + 'angle_video.mp4')  # path to video
    tc.meanshift(Rectangle(600, 400, width=100, height=100))  # set rectangle
