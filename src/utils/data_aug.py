import os
import json
import imageio
import cv2
import imgaug as ia
import numpy as np
from imgaug import augmenters as iaa
from imgaug.augmentables.polys import Polygon, PolygonsOnImage
from all_paths import Paths

paths = Paths('')


class Augmentator(object):
    def __init__(self):
        pass

    def generate(self, path_to_pics, path_to_ann, path_to_save):
        annotations = json.load(open(path_to_ann))
        annotations = list(annotations.values())
        annotations = [a for a in annotations if a['regions']]
        json_save = os.path.join(path_to_save, 'ann.json')
        json_all = {}
        json_all_arr = []
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
                json_all_arr.append((f'{file_name}{i}', json_cur))
                saved_path = os.path.join(path_to_save, file_name)
                self.save_picture(img, saved_path)
                print('Saved in ', saved_path)
                i += 2
        json_train, json_val = self.div_arr(json_all_arr)
        with open(os.path.join(path_to_save, 'ann_train.json'), 'w', encoding='utf-8') as f:
            json.dump(json_train, f, ensure_ascii=False, indent=1)
        with open(os.path.join(path_to_save, 'ann_val.json'), 'w', encoding='utf-8') as f:
            json.dump(json_val, f, ensure_ascii=False, indent=1)
        with open(json_save, 'w', encoding='utf-8') as f:
            json.dump(json_all, f, ensure_ascii=False, indent=1)

    def div_arr(self, arr):
        np.random.permutation(arr)
        part = 0.7 * len(arr)
        json_1 = {}
        json_2 = {}
        for i, node in enumerate(arr):
            if i < part:
                json_1[node[0]] = node[1]
            else:
                json_2[node[0]] = node[1]
        return json_1, json_2

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

        for i in range(-30, 30, 10):
            aug_func = iaa.Sequential([
                iaa.Affine(rotate=i)
            ])
            image_aug, psoi_aug = aug_func(image=image, polygons=psoi)
            aug_images.append(image_aug)
            psoi_augs.append(psoi_aug)
            # images = [psoi_aug.draw_on_image(image_aug, alpha_face=0.2, size_points=7), image]
            # ia.imshow(np.hstack(images))

        aug_func = iaa.Sequential([
            iaa.Fliplr(1)
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
    aug.generate(paths.IMAGES_PATH + 'all_pics/', paths.IMAGES_PATH + 'all_pics/all_pics.json', paths.IMAGES_PATH + 'tmp/')
