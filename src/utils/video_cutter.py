import os
import time
import cv2
import numpy as np
from matplotlib import pyplot as plt

class Frame(object):
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

class Cutter(object):
    def __init__(self, path_to_video, path_to_pic, pic_name, frames):
        self.path_to_pic = path_to_pic
        self.pic_name = pic_name
        self.path_to_video = path_to_video
        self.frames = frames

    def run(self):
        video_cap = cv2.VideoCapture(self.path_to_video)
        #flag = 1
        frame_number = 0
        while video_cap.isOpened():
            success, frame = video_cap.read()
            if not success:
                break
            #if flag == 1:
            for i in self.frames:
                cv2.rectangle(frame, (i.x, i.y), (i.x + i.width, i.y + i.height), (255, 0, 0), 2)
                #flag = 0
            cv2.imshow('video', frame)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame,'OK',(10,500), font, 4,(0,0,255),2,cv2.LINE_AA)
                cv2.imshow('video', frame)
                for i in self.frames:
                    cut_frame = cut_frame = frame[i.y:i.y+i.height, i.x:i.x+i.width]
                    cv2.imwrite(os.path.join(self.path_to_pic, self.pic_name + str(frame_number) + '.jpg'), cut_frame)
                    print(f"Save: {os.path.join(self.path_to_pic, self.pic_name + str(frame_number) + '.jpg')}")
                    frame_number += 1
                time.sleep(5)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        video_cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    #frames = [Frame(480, 130, 700, 480)] vid1
    frames = [Frame(551, 337, 902 - 551, 603 - 337), Frame(908, 337, 1260 - 908, 603 - 337)]
    cutter = Cutter('data/videos/vid2.mp4', 'data/images/tmp4', 'pic', frames)
    cutter.run()
