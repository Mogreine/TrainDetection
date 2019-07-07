import os
import json
import imageio
import cv2
import imgaug as ia
import numpy as np
from imgaug import augmenters as iaa
from imgaug.augmentables.polys import Polygon, PolygonsOnImage


class Augmentator(object):
    seq2 = iaa.Sequential([
        iaa.CropAndPad(percent=(-0.2, 0.2), pad_mode="edge"),
        iaa.AddToHueAndSaturation((-60, 60)),
        iaa.ElasticTransformation(alpha=90, sigma=9),
        iaa.CoarseDropout((0.01, 0.1), size_percent=0.01)
    ], random_order=True)

    brightness_change = iaa.Sequential([
        iaa.Multiply([0.2, 1, 1.5])
    ])

    def __init__(self):
        pass

    def generate(self, path_to_pics, path_to_ann, path_to_save):
        annotations = json.load(open(path_to_ann))
        annotations = list(annotations.values())
        annotations = [a for a in annotations if a['regions']]
        for a in annotations:
            image_path = os.path.join(path_to_pics, a['filename'])
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]
            file_seg = a['filename'].split('.')
            imges = self.generate_images(image_path, polygons)
            i = 1
            for img in imges:
                saved_path = os.path.join(path_to_save, f"{file_seg[0]}_{i}.{file_seg[1]}")
                self.save_picture(img, saved_path)
                print('Saved in ', saved_path)
                i += 2

    def generate_images(self, path_to_pic, shapes):
        image = imageio.imread(path_to_pic)
        pols = []
        for p in shapes:
            if p['name'] == 'rect':
                x = p['x']
                y = p['y']
                width = p['width']
                height = p['height']
                pol = Polygon([
                    (x, y),
                    (x + width, y),
                    (x + width, y + height),
                    (x, y + height)
                ])
                pols.append(pol)
            else:
                x_all = p['all_points_x']
                y_all = p['all_points_y']
                points = []
                for i in range(len(x_all)):
                    points.append((x_all[i], y_all[i]))
                pol = Polygon(points)
                pols.append(pol)
        psoi = ia.PolygonsOnImage(pols, shape=image.shape)
        aug_images = []
        for i in range(1, 16, 2):
            aug_func = iaa.Sequential([
                iaa.Multiply(i / 10)
            ])
            image_aug, psoi_aug = aug_func(image=image, polygons=psoi)
            aug_images.append(image_aug)
            # imageio.save(save_path_name)
            # return psoi_aug
            images = [psoi_aug.draw_on_image(image_aug, alpha_face=0.2, size_points=7), image]
            ia.imshow(np.hstack(images))
        return aug_images

    def save_picture(self, img, file_name):
        cv2.imwrite(file_name, img)


if __name__ == "__main__":
    ia.seed(4)
    aug = Augmentator()
    aug.generate('../../data/images/train', '../../data/via/new/train_plates_rect.json', '../../data/images/aug_train')
    # aug.test_proc('test.jpg')
