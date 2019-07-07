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
        json_save = os.path.join(path_to_save, 'ann.json')
        json_all = {}
        for a in annotations:
            image_path = os.path.join(path_to_pics, a['filename'])
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
                reg_attr = [r['region_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]
                reg_attr = [r['region_attributes'] for r in a['regions']]
            file_seg = a['filename'].split('.')

            imges, psois = self.generate_images(image_path, polygons)
            i = 1
            for img, psoi in zip(imges, psois):
                json_cur = {}
                file_name = f"{file_seg[0]}_{i}.{file_seg[1]}"
                json_cur['filename'] = file_name
                json_cur['size'] = a['size']
                json_cur['regions'] = self.get_regions(psoi, reg_attr)
                json_cur['file_attributes'] = {}
                json_all[f'{file_name}{i}'] = json_cur
                saved_path = os.path.join(path_to_save, file_name)
                self.save_picture(img, saved_path)
                print('Saved in ', saved_path)
                i += 2
        with open(json_save, 'w', encoding='utf-8') as f:
            json.dump(json_all, f, ensure_ascii=False, indent=1)

    def get_regions(self, psoi_aug, reg_attr):
        regions = []
        for i, p in enumerate(psoi_aug.polygons):
            region = {}
            x_all = p.xx_int.tolist()
            y_all = p.yy_int.tolist()
            region['shape_attributes'] = {'name': 'polygon', 'all_points_x': x_all, 'all_points_y': y_all}
            region['region_attributes'] = reg_attr[i]
            regions.append(region)
        return regions

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
        psoi_augs = []
        for i in range(1, 16, 2):
            aug_func = iaa.Sequential([
                iaa.Multiply(i / 10)
            ])
            image_aug, psoi_aug = aug_func(image=image, polygons=psoi)
            aug_images.append(image_aug)
            psoi_augs.append(psoi_aug)
            # images = [psoi_aug.draw_on_image(image_aug, alpha_face=0.2, size_points=7), image]
            # ia.imshow(np.hstack(images))
        return aug_images, psoi_augs

    def save_picture(self, img, file_name):
        cv2.imwrite(file_name, img)


if __name__ == "__main__":
    ia.seed(4)
    aug = Augmentator()
    # aug.generate('data/images/val', 'data/via/new/test_plates_polygon.json', 'data/images/test')
    aug.generate('../../data/images/all_pics', '../../data/via/via_export_json.json', '../../data/images/aug_all')
    # aug.test_proc('test.jpg')
