import cv2
import os
from src.utils.all_paths import Paths

paths = Paths('../../')


class FrameCutter:
    def __init__(self, _video_path=paths.VIDEOS_PATH+'test_video.mp4',
                 _pics_path=paths.IMAGES_PATH+'tmp/',
                 _pic_name='pic'):
        self.video_path = _video_path
        self.pics_path = _pics_path
        self.pic_name = _pic_name

    def cut(self, frame_count, x, y, height, width):
        self.delete_files()
        video_cap = cv2.VideoCapture(self.video_path)
        frame_number = 0
        curr_frame = 0
        while video_cap.isOpened():
            success, frame = video_cap.read()
            if not success:
                break

            if curr_frame < frame_count:
                curr_frame += 1
                continue
            curr_frame = 0

            cut_frame = frame[y:y+height, x:x+width]

            cv2.imwrite(os.path.join(self.pics_path, self.pic_name + str(frame_number) + '.jpg'), cut_frame)
            cv2.imshow('Video', cut_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            print('Save: ' + self.pic_name + str(frame_number) + '.jpg')
            frame_number += 1
        video_cap.release()
        cv2.destroyAllWindows()

    def delete_files(self):
        for file in os.listdir(self.pics_path):
            file_path = os.path.join(self.pics_path, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)
