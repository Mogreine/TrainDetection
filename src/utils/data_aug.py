import os
import json
import imageio
import cv2
import imgaug as ia
import numpy as np
from imgaug import augmenters as iaa
from imgaug.augmentables.polys import Polygon, PolygonsOnImage
from src.utils.all_paths import Paths
from typing import List, Tuple, Dict

paths = Paths('../../')


class Augmentator(object):
    def __init__(self):
        pass

    def generate(self, path_to_pics: str, path_to_ann: str, path_to_save: str) -> None:
        annotations = json.load(open(path_to_ann))
        annotations = list(annotations.values())
        annotations = [a for a in annotations if a['regions']]
        json_save = os.path.join(path_to_save, 'all.json')
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
                saved_path = os.path.join(path_to_save, file_name)
                pic_size = self.save_picture(img, saved_path)
                json_all[f'{file_name}{pic_size}'] = json_cur
                json_all_arr.append((f'{file_name}{pic_size}', json_cur))
                print('Saved in ', saved_path)
                i += 1
        json_train, json_val = self.split_arr(json_all_arr)
        with open(os.path.join(path_to_save, 'ann_train.json'), 'w', encoding='utf-8') as f:
            json.dump(json_train, f, ensure_ascii=False, indent=1)

        with open(os.path.join(path_to_save, 'ann_val.json'), 'w', encoding='utf-8') as f:
            json.dump(json_val, f, ensure_ascii=False, indent=1)

        with open(json_save, 'w', encoding='utf-8') as f:
            json.dump(json_all, f, ensure_ascii=False, indent=1)

    def split_arr(self, arr: List, train_part: float = 0.7) -> (Dict, Dict):
        arr = np.random.permutation(arr)
        train_size = train_part * len(arr)
        train_json = {}
        val_json = {}
        for i, node in enumerate(arr):
            if i <= train_size:
                train_json[node[0]] = node[1]
            else:
                val_json[node[0]] = node[1]
        return train_json, val_json

    def get_regions(self, psoi_aug, reg_attr) -> List[Dict]:
        regions = []
        height, width, _ = psoi_aug.shape
        for i, p in enumerate(psoi_aug.polygons):
            region = {}
            x_all = p.xx_int.tolist()
            y_all = p.yy_int.tolist()
            for z in range(len(x_all)):
                if x_all[z] < 0:
                    x_all[z] = 0
                elif x_all[z] >= width:
                    x_all[z] = width - 1
            for z in range(len(y_all)):
                if y_all[z] < 0:
                    y_all[z] = 0
                elif y_all[z] >= height:
                    y_all[z] = height - 1
            region['shape_attributes'] = {'name': 'polygon', 'all_points_x': x_all, 'all_points_y': y_all}
            region['region_attributes'] = reg_attr[i]
            regions.append(region)
        return regions

    def generate_images(self, path_to_pic: str, shapes: List[Dict]) -> (List, List):
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
        transforms = [iaa.Fliplr(1),
                      iaa.GaussianBlur(2),
                      iaa.SaltAndPepper(0.2),
                      iaa.Affine(rotate=-10),
                      iaa.Affine(rotate=10),
                      iaa.Multiply(0.6),
                      iaa.Multiply(0.8),
                      iaa.Multiply(1.2),
                      iaa.ElasticTransformation(sigma=0.2, alpha=3)]
        for trans in transforms:
            aug_func = iaa.Sequential([
                trans
            ])
            image_aug, psoi_aug = aug_func(image=image, polygons=psoi)
            aug_images.append(image_aug)
            psoi_augs.append(psoi_aug)
        # images = [psoi_aug.draw_on_image(image_aug, alpha_face=0.2, size_points=7), image]
        # ia.imshow(np.hstack(images))
        return aug_images, psoi_augs

    def save_picture(self, img: np.ndarray, file_name: str) -> int:
        cv2.imwrite(file_name, img)
        return os.path.getsize(file_name)


if __name__ == "__main__":
    ia.seed(4)
    aug = Augmentator()
    aug.generate(paths.IMAGES_PATH + 'numbers/', paths.IMAGES_PATH + 'numbers/fixed_ones.json',
                 paths.IMAGES_PATH + 'tmp/')
