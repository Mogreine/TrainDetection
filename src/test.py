import cv2
import os
import json
import matplotlib as plt

def get_average(path_to_pics, path_to_ann):
    annotations = json.load(open(path_to_ann))
    annotations = list(annotations.values())
    annotations = [a for a in annotations if a['regions']]
    width, height = 0, 0
    count = 0
    for a in annotations:
        if type(a['regions']) is dict:
            polygons = [r['shape_attributes'] for r in a['regions'].values()]
        else:
            polygons = [r['shape_attributes'] for r in a['regions']]
        for p in polygons:
            if p['name'] == 'rect':
                width += p['width']
                height += p['height']
            else:
                pass
            count += 1
    return width / count, height / count


def test(path_to_pics, names):
    for n in names:
        img = cv2.imread(os.path.join(path_to_pics, n))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.Canny(img_gray, 20, 30)
        cv2.imshow('images', img)
        while cv2.waitKey(1) & 0xFF != ord('q'):
            pass


def surf(path_to_pics, names):
    for n in names:
        pic_path = os.path.join(path_to_pics, n)
        img = cv2.imread(pic_path, 0)
        # surf = cv2.SURF(400)
        surf = cv2.xfeatures2d.SURF_create(400)
        kp, des = surf.detectAndCompute(img,None)
        surf.setHessianThreshold(5000)
        kp, des = surf.detectAndCompute(img,None)
        img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
        cv2.imshow('img', img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # width, height = get_average('data/images', 'data/images/front_pics/ann.json')
    # print(f"av width: {width}, av height: {height}")
    surf('data/images/tmp2', ['1.jpg', '2.jpg', '3.jpg'])