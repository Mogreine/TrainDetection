import cv2
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':

    video = cv2.VideoCapture('../data/videos/test_video.mp4')

    while video.isOpened():
        success, frame = video.read()
        if not success:
            break

        img = frame
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(img, 60, 120)
        cv2.imshow('test', edges)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

