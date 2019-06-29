import cv2
import os

def cut(video_path, frame_count, pics_path, name):
    video_cap = cv2.VideoCapture(video_path)
    frame_current = 0
    frame_number = 0
    while video_cap.isOpened():
        success, frame = video_cap.read()
        if not success:
            break
        if frame_current < frame_count:
            frame_current += 1
            continue
        frame_current = 0
        cv2.imwrite(os.path.join(pics_path, name + str(frame_number) + '.jpg'), frame)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        print('Save: ' + name + str(frame_number) + '.jpg')
        frame_number += 1
    video_cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    cut('test.mp4', 120, 'G:\work\ML\TrainDetection\pics', 'train_test')
