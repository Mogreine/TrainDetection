import os
import time
import cv2
import numpy as np
from matplotlib import pyplot as plt
from src.utils.all_paths import Paths

paths = Paths('../')


class Rectangle(object):
    """ Represents a rectangle
    """

    def __init__(self, x: int, y: int, width: int = None, height: int = None, x2: int = None, y2: int = None):
        """
        Input:
            x, y - coordinates of the left upper corner
            width, height - width and height of the rectangle
            or
            x2, y2 - coordinates of the right bottom corner
        """
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

    def get_points(self) -> (int, int, int, int):
        """
        Return:
            x1, y1, x2, y2 - coordinates of left upper and right bottom corners of the rectangle
        """
        return self.lu_x, self.lu_y, self.rd_x, self.rd_y

    def get_with_params(self) -> (int, int, int, int):
        """
        Return:
            x, y - coordinates of left upper of the rectangle
            width, height - width and height of the rectangle
        """
        return self.lu_x, self.lu_y, self.width, self.height


class TrainCounter(object):
    """ Train counting
    Params:
        train_count - number of trains already passed
        threshold - distance between tracking the window and a video's screen border
    """
    train_count = 0
    threshold = 20

    def __init__(self, path_to_video: str, path_to_save: str = None):
        """ Constuctor
            Params:
                path_to_video - path to video file
                path_to_save - path to save file if need
        """
        self.path_to_video = path_to_video
        self.path_to_save = path_to_save

    def camshift(self, rect: Rectangle) -> None:
        """ Implements camshift algorithm from OpenCV lib
        """

        # take the first frame
        capture = cv2.VideoCapture(self.path_to_video)
        ret, frame = capture.read()

        # set up the init location of the window
        track_window = rect.get_with_params()

        # set up ROI for tracking
        roi = frame[rect.lu_y:rect.lu_y + rect.height,
                    rect.lu_x:rect.lu_x + rect.width]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)),
                           np.array((180., 255., 255.)))
        roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        # set up a termination criteria, either 10 iteration or move by at least 1 pt
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

        # update X coordinate of the upper-left corner of the tracking window
        rect.lu_x += 200

        while capture.isOpened():
            ret, frame = capture.read()
            if not ret:
                break
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

            # apply CamShift to get a new location
            ret, track_window = cv2.CamShift(dst, track_window, term_crit)

            # get a min point of the tracking window
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            mm = pts.min(axis=0)[0]

            # if dist <= threshold increment train_count and set up a new tracking window
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
            
            # draw rect on the screen
            img2 = cv2.polylines(frame, [pts], True, 255, 2)
            cv2.imshow('img2', img2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        capture.release()
        cv2.destroyAllWindows()

    def meanshift(self, rect: Rectangle) -> None:
        """ Implements meanshift algorithm from OpenCV lib
        """
        capture = cv2.VideoCapture(self.path_to_video)

        # if necessary, setup VideoWriter
        if self.path_to_save is not None:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(capture.get(cv2.CAP_PROP_FPS))
            vwriter = cv2.VideoWriter(self.path_to_save,
                                      cv2.VideoWriter_fourcc(*'MJPG'),
                                      fps, (width, height))
        
        # take the first frame
        ret, frame = capture.read()

        # set up the init location of the window
        track_window = rect.get_with_params()

        # set up ROI for tracking
        roi = frame[rect.lu_y:rect.lu_y + rect.height,
                    rect.lu_x:rect.lu_x + rect.width]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)),
                           np.array((180., 255., 255.)))
        roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(roi_hist, roi_hist, 0, 255,
                      cv2.NORM_MINMAX)

        # set up a termination criteria, either 10 iteration or move by at least 1 pt
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

        # update X coordinate of the upper-left corner of the tracking window
        rect.lu_x += 100

        # set up font to show the number of passed trains on the screen
        font = cv2.FONT_HERSHEY_SIMPLEX

        while capture.isOpened():
            ret, frame = capture.read()
            if not ret:
                break
            
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

            # apply meanshift to get a new location
            ret, track_window = cv2.meanShift(dst, track_window, term_crit)
            x, y, w, h = track_window
            mm = x
            
            # if dist <= threshold increment train_count and set up a new tracking window
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
            
            # add tracking window on the image
            img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)

            # add train_count on the image; write the frame on disk
            if self.path_to_save is not None:
                img2 = cv2.putText(img2, str(self.train_count),
                                   (10, 500), font, 4, (255, 0, 0), 2, cv2.LINE_AA)
                vwriter.write(img2)
            
            cv2.imshow('img2', img2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        capture.release()
        if self.path_to_save is not None:
            vwriter.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tc = TrainCounter(paths.VIDEOS_PATH + 'angle_video.mp4')
    # set up a rectangle
    tc.meanshift(Rectangle(600, 400, width=100, height=100))
