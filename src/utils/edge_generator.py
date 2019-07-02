import cv2
import numpy as np
from matplotlib import pyplot as plt
from frame_cutter import FrameCutter

if __name__ == '__main__':
    cutter = FrameCutter()
    cutter.cut(200, 300, 400, 200, 300)