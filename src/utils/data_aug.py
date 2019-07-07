import os
import json
import imageio
import imgaug as ia
import numpy as np
from imgaug import augmenters as iaa
from imgaug.augmentables.polys import Polygon, PolygonsOnImage


class Augmentator(object):
    seq = iaa.Sequential([
            iaa.Affine(rotate=(-25, 25)),
            iaa.AdditiveGaussianNoise(scale=(10, 60)),
            iaa.Crop(percent=(0, 0.2))
        ], random_order=True)

    seq2 = iaa.Sequential([
        iaa.CropAndPad(percent=(-0.2, 0.2), pad_mode="edge"),
        iaa.AddToHueAndSaturation((-60, 60)),
        iaa.ElasticTransformation(alpha=90, sigma=9),
        iaa.CoarseDropout((0.01, 0.1), size_percent=0.01)
    ], random_order=True)

    mix = iaa.Sequential([
        iaa.Affine(rotate=(-25, 25)),
        iaa.AdditiveGaussianNoise(scale=(10, 60)),
        iaa.Crop(percent=(0, 0.2)),
        iaa.CropAndPad(percent=(-0.2, 0.2), pad_mode="edge"),
        iaa.AddToHueAndSaturation((-60, 60)),
        iaa.ElasticTransformation(alpha=90, sigma=9),
        iaa.CoarseDropout((0.01, 0.1), size_percent=0.01)
    ], random_order=True)

    def __init__(self):
        pass
    
    def proc_all(self, path_to_pics, path_to_ann, path_to_save, count_from_pic):
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
            for i in range(count_from_pic):
                saved_path = os.path.join(path_to_save, f"{file_seg[0]}_{i}.{file_seg[1]}")
                psoi_aug = self.proc(image_path, polygons, saved_path)
                print('Saved in ', saved_path)
                #print(polygons)

    def proc(self, path_to_pic, shapes, save_path_name):
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
        image_aug, psoi_aug = self.seq2(image=image, polygons=psoi)
        # imageio.save(save_path_name)
        # return psoi_aug
        images = [psoi_aug.draw_on_image(image_aug, alpha_face=0.2, size_points=7), image]
        ia.imshow(np.hstack(images))
        return 0

    def test_proc(self, path_to_pic):
        image = imageio.imread(path_to_pic)
        images = [image, image, image, image, image, image, image, image]
        rotate = iaa.Affine(rotate=(-25, 25))
        

        images_aug = self.mix.augment_images(images)
        ia.imshow(np.hstack(images_aug))

if __name__ == "__main__":
    ia.seed(4)
    aug = Augmentator()
    aug.proc_all('data/images/val', 'data/via/new/test_plates_polygon.json', 'data/images/test', 1)
    #aug.test_proc('test.jpg')