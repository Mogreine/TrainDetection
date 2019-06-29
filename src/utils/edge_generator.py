import cv2
import numpy as np
from matplotlib import pyplot as plt
from src.utils.frame_cutter import FrameCutter

if __name__ == '__main__':
    cutter = FrameCutter()
    cutter.cut(380, 300, 400, 200, 300)

