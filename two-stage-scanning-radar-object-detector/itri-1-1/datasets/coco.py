import logging
from glob import glob
from pathlib import Path

import cv2
import numpy as np
from imgaug import augmenters as iaa
from imgaug.augmentables import BoundingBox, BoundingBoxesOnImage, Keypoint, KeypointsOnImage
from omegaconf.listconfig import ListConfig
from PIL import Image
from pycocotools.coco import COCO
from torch.utils import data

from utils.helper import instantiate_augmenters
from utils.image import draw_umich_gaussian as draw_gaussian
from utils.image import gaussian_radius
from utils.box import get_annotation_with_angle, rotate_bbox

cv2.setNumThreads(0)
log = logging.getLogger(__name__)


class Dataset(data.Dataset):
    def __init__(
            self, image_folder, annotation_file, input_size=(512, 512),
            target_domain_glob=None, num_classes=80, num_keypoints=0,
            rotated_boxes=False, mean=(0.40789654, 0.44719302, 0.47026115),
            std=(0.28863828, 0.27408164, 0.27809835),
            augmentation=None, augment_target_domain=False, max_detections=150,
            down_ratio=4):
        self.image_folder = Path(image_folder)
        self.coco = COCO(annotation_file)
        self.images = self.coco.getImgIds()
        self.use_rotated_boxes = rotated_boxes
        self.max_detections = max_detections
        self.down_ratio = down_ratio
        self.input_size = input_size
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        self.augmentation = augmentation
        self.num_classes = num_classes
        self.num_keypoints = num_keypoints
        self.string_id_mapping = {}
        self.augment_target_domain = augment_target_domain
        # self.x_y_theta_range = self.ret_x_y_range()
        self.cat_mapping = {v: i for i,
                            v in enumerate(range(1, num_classes + 1))}
        self.classes = {y: self.coco.cats[x] if x in self.coco.cats else ''
                        for x, y in self.cat_mapping.items()}
        assert len(input_size) == 2

        if isinstance(target_domain_glob, str):
            self.target_domain_files = glob(target_domain_glob)
        elif isinstance(target_domain_glob, (list, ListConfig)):
            self.target_domain_files = []
            for pattern in target_domain_glob:
                self.target_domain_files.extend(glob(pattern))
        else:
            self.target_domain_files = []

        if self.augmentation:
            augmentation_methods = instantiate_augmenters(augmentation)
            self.augmentation = iaa.Sequential(augmentation_methods)

        self.resize = iaa.Resize((self.input_size[0], self.input_size[1]))
        self.resize_out = iaa.Resize(
            (self.input_size[0] // down_ratio,
             self.input_size[1] // down_ratio))

        log.info(
            f"found {len(self.target_domain_files)} samples for target domain")
        super().__init__()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = self.image_folder / file_name
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_detections)
        img = np.array(Image.open(img_path).convert("RGB"))

        if self.use_rotated_boxes:
            ret = self.__get_rotated_coco(img, anns, num_objs)
        else:
            ret = self.__get_default_coco(img, anns, num_objs)

        if isinstance(img_id, str):
            mapped_id = self.string_id_mapping.get(
                img_id, 1 + len(self.string_id_mapping))
            self.string_id_mapping[img_id] = mapped_id
            img_id = mapped_id

        ret['id'] = img_id

        if len(self.target_domain_files):
            target_domain_img = np.array(Image.open(
                np.random.choice(self.target_domain_files)).convert("RGB"))

            if self.augmentation is not None and self.augment_target_domain:
                target_domain_img = self.augmentation(image=target_domain_img)

            target_domain_img = self.resize(image=target_domain_img)
            target_domain_img = np.array(
                target_domain_img, dtype=np.float32) / 255.0
            target_domain_img = (target_domain_img - self.mean) / self.std
            target_domain_img = target_domain_img.transpose(2, 0, 1)

            ret['target_domain_input'] = target_domain_img

        return ret

    # sungting
    def ret_normalize(self,x):
        return (x - np.mean(x)) / (np.std(x))

    # sungting
    def ret_x_y_range(self):
        shape = (1152, 1152)

        x = np.linspace(-100, 100, shape[1])
        y = np.linspace(100, -100, shape[1])

        X, Y = np.meshgrid(x, y)

        ran = np.sqrt(X**2 + Y**2)
        ang = np.arctan2(X, Y) % (2*np.pi)

        return list(map(self.ret_normalize, (X, Y, ang, ran)))

    # sungting
    def combine_int_x_y_theta_ran(self, img):
        return np.dstack((img[:,:,0]))
                        #   self.x_y_theta_range[0],
                        #   self.x_y_theta_range[1],
                        #   self.x_y_theta_range[2],
                        #   self.x_y_theta_range[3]))

    def __get_default_coco(self, img, anns, num_objs):
        boxes = []
        if self.num_keypoints > 0:
            kpts = []

        for k in range(num_objs):
            ann = anns[k]
            bbox = self._coco_box_to_bbox(ann['bbox'])
            boxes.append(BoundingBox(*bbox))

            if self.num_keypoints > 0:
                if 'keypoints' not in ann:
                    ann['keypoints'] = np.zeros((3 * self.num_keypoints,))

                kpt = [
                    Keypoint(*x)
                    for x in np.array(ann['keypoints']).reshape(-1, 3)
                    [:, : 2]]
                kpts.extend(kpt)

        bbs = BoundingBoxesOnImage(boxes, shape=img.shape)

        if self.num_keypoints > 0:
            kpts = KeypointsOnImage(kpts, shape=img.shape)

        if self.augmentation is not None:
            if self.num_keypoints > 0:
                img_aug, bbs_aug, kpts_aug = self.augmentation(
                    image=img, bounding_boxes=bbs, keypoints=kpts)
            else:
                # import ipdb; ipdb.set_trace(context=7)
                img_aug, bbs_aug = self.augmentation(
                    image=img, bounding_boxes=bbs)
        else:
            if self.num_keypoints > 0:
                kpts_aug = kpts.copy()


            img_aug, bbs_aug = np.copy(img), bbs.copy()

        if self.num_keypoints > 0:
            img_aug, bbs_aug, kpts_aug = self.resize(
                image=img_aug, bounding_boxes=bbs_aug, keypoints=kpts_aug)
        else:
            # import ipdb; ipdb.set_trace(context=7)
            
            img_aug, bbs_aug = self.resize(
                image=img_aug, bounding_boxes=bbs_aug)

        # import ipdb; ipdb.set_trace(context=7)
        
        img = (img_aug.astype(np.float32) / 255.)
        inp = (img - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)

        output_h = self.input_size[1] // self.down_ratio
        output_w = self.input_size[0] // self.down_ratio
        num_classes = self.num_classes

        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_detections, 2), dtype=np.float32)
        reg = np.zeros((self.max_detections, 2), dtype=np.float32)
        ind = np.zeros((self.max_detections), dtype=np.int64)
        reg_mask = np.zeros((self.max_detections), dtype=np.uint8)
        gt_det = np.zeros((self.max_detections, 6), dtype=np.float32)
        gt_areas = np.zeros((self.max_detections), dtype=np.float32)

        if self.num_keypoints > 0:
            kp = np.zeros(
                (self.max_detections,
                 self.num_keypoints * 2),
                dtype=np.float32)
            gt_kp = np.zeros(
                (self.max_detections, self.num_keypoints, 2), dtype=np.float32)
            kp_reg_mask = np.zeros(
                (self.max_detections, self.num_keypoints * 2), dtype=np.uint8)

            bbs_aug, kpts_aug = self.resize_out(
                bounding_boxes=bbs_aug, keypoints=kpts_aug)
        else:
            bbs_aug = self.resize_out(bounding_boxes=bbs_aug)

        for k in range(num_objs):
            ann = anns[k]
            bbox_aug = bbs_aug[k].clip_out_of_image((output_w, output_h))
            bbox = np.array([bbox_aug.x1, bbox_aug.y1,
                             bbox_aug.x2, bbox_aug.y2])

            cls_id = int(self.cat_mapping[ann['category_id']])

            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                radius = gaussian_radius((np.ceil(h), np.ceil(w)))
                radius = max(0, int(radius))
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
                    dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_gaussian(hm[cls_id], ct_int, radius)
                wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1
                gt_det[k] = ([ct[0] - w / 2, ct[1] - h / 2,
                              ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])

                if self.num_keypoints > 0:
                    valid = np.array(ann["keypoints"]).reshape(-1, 3)[:, -1]
                    for i, p in enumerate(
                            kpts_aug[k * self.num_keypoints: k * self.num_keypoints + self.num_keypoints]):
                        kp[k][i * 2] = p.x - ct_int[0]
                        kp[k][i * 2 + 1] = p.y - ct_int[1]

                        is_valid = valid[i] == 2 and not p.is_out_of_image(
                            (output_w, output_w))
                        kp_reg_mask[k, i * 2] = int(is_valid)
                        kp_reg_mask[k, i * 2 + 1] = int(is_valid)
                        gt_kp[k][i] = p.x, p.y

                if "area" not in ann:
                    gt_areas[k] = w * h
                else:
                    gt_areas[k] = ann["area"]

        del bbs
        del bbs_aug
        del img_aug

        gt_det = np.array(gt_det, dtype=np.float32) if len(
            gt_det) > 0 else np.zeros((1, 6), dtype=np.float32)

        ret = {
            'input': inp,
            'hm': hm,
            'reg_mask': reg_mask,
            'ind': ind,
            'wh': wh,
            'reg': reg,
            'gt_dets': gt_det,
            'gt_areas': gt_areas,
        }

        if self.num_keypoints > 0:
            ret['kps'] = kp
            ret['gt_kps'] = gt_kp
            ret['kp_reg_mask'] = kp_reg_mask
            del kpts_aug

        return ret

    def __get_rotated_coco(self, img, anns, num_objs):
        # img = self.combine_int_x_y_theta_ran(img)
        # import ipdb; ipdb.set_trace(context=7)
        
        kpts = []
        kpts_tmp = []
        for k in range(num_objs):
            ann = anns[k]
            ann_rotated = get_annotation_with_angle(ann) # get cx,cy,w,h,angle
            ann_rotated[4] = ann_rotated[4]
            rot = rotate_bbox(*ann_rotated) # get bbox four corners
            kpts.extend([Keypoint(*x) for x in rot])

            if self.num_keypoints > 0:
                if 'keypoints' not in ann:
                    ann['keypoints'] = np.zeros((3 * self.num_keypoints,))

                kpt = [
                    Keypoint(*x)
                    for x in np.array(ann['keypoints']).reshape(-1, 3)
                    [:, : 2]]
                kpts_tmp.extend(kpt)

        idx_boxes = len(kpts)
        if self.num_keypoints > 0:
            kpts.extend(kpts_tmp)
        
        kpts = KeypointsOnImage(kpts, shape=img.shape)

        if self.augmentation is not None:
            img_aug, kpts_aug = self.augmentation(image=img, keypoints=kpts)
        else:
            img_aug, kpts_aug = np.copy(img), kpts.copy()

        img_aug, kpts_aug = self.resize(
            image=img_aug, keypoints=kpts_aug)
        
        # sungting
        # if True:
        #     np_img = np.dstack((img_aug[..., 0],)*3).astype(np.uint8)
        #     kp_img = kpts_aug.draw_on_image(image=np_img)
            

        # img = (img_aug.astype(np.float32) / 255.)
        # inp = (img - self.mean) / self.std
        # inp = inp.transpose(2, 0, 1)

        img_aug = img_aug.astype(np.float32)
        img_aug[:, :, 0] = img_aug[:, :, 0] / 255.
        img_aug[:, :, 0] = (img_aug[:, :, 0] - self.mean[0, 0, 0]) / self.std[0, 0, 0]
        inp = img_aug.transpose(2, 0, 1)

        output_h = self.input_size[1] // self.down_ratio
        output_w = self.input_size[0] // self.down_ratio
        num_classes = self.num_classes

        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_detections, 3), dtype=np.float32)
        reg = np.zeros((self.max_detections, 2), dtype=np.float32)
        ind = np.zeros((self.max_detections), dtype=np.int64)
        reg_mask = np.zeros((self.max_detections), dtype=np.uint8)
        gt_det = np.zeros(
            (self.max_detections,
             7 if self.use_rotated_boxes else 6),
            dtype=np.float32)
        gt_areas = np.zeros((self.max_detections), dtype=np.float32)

        if self.num_keypoints > 0:
            kp = np.zeros(
                (self.max_detections,
                 self.num_keypoints * 2),
                dtype=np.float32)
            gt_kp = np.zeros(
                (self.max_detections, self.num_keypoints, 2), dtype=np.float32)
            kp_reg_mask = np.zeros(
                (self.max_detections, self.num_keypoints * 2), dtype=np.uint8)

        kpts_aug = self.resize_out(keypoints=kpts_aug)

        box_kpts_aug, kpts_aug = kpts_aug[:idx_boxes], kpts_aug[idx_boxes:]
        assert num_objs == len(box_kpts_aug) // 4

        for k in range(num_objs):
            ann = anns[k]
            points = []
            for p in box_kpts_aug[k * 4: k * 4 + 4]:
                box_kp = list((np.clip(p.x, 0, output_w - 1),
                               np.clip(p.y, 0, output_h - 1)))
                points.append(box_kp)

            points = np.array(points).astype(np.float32)
            cv_ct, cv_wh, cv_angle = cv2.minAreaRect(points)

            if cv_wh[0] == 0 or cv_wh[1] == 0:
                continue

            cx, cy, w, h, angle = get_annotation_with_angle({'rbbox': np.array(
                [cv_ct[0], cv_ct[1], cv_wh[0], cv_wh[1], cv_angle])})

            # sungting
            angle += 90

            ct = np.array((cx, cy))

            cls_id = int(self.cat_mapping[ann['category_id']])

            if h > 0 and w > 0:
                radius = gaussian_radius((np.ceil(h), np.ceil(w)))
                radius = max(0, int(radius))
                ct_int = ct.astype(np.int32)
                draw_gaussian(hm[cls_id], ct_int, radius)
                wh[k] = w, h, angle
                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1
                gt_det[k] = ([ct[0], ct[1], w, h, angle, 1, cls_id])

                if self.num_keypoints > 0:
                    valid = np.array(ann["keypoints"]).reshape(-1, 3)[:, -1]
                    for i, p in enumerate(
                            kpts_aug[k * self.num_keypoints: k * self.num_keypoints + self.num_keypoints]):
                        kp[k][i * 2] = p.x - ct_int[0]
                        kp[k][i * 2 + 1] = p.y - ct_int[1]

                        is_valid = valid[i] == 2 and not p.is_out_of_image(
                            (output_w, output_w))
                        kp_reg_mask[k, i * 2] = int(is_valid)
                        kp_reg_mask[k, i * 2 + 1] = int(is_valid)
                        gt_kp[k][i] = p.x, p.y

                if "area" not in ann:
                    gt_areas[k] = w * h
                else:
                    gt_areas[k] = ann["area"]

        del box_kpts_aug
        del img_aug

            


        gt_det = np.array(gt_det, dtype=np.float32) if len(
            gt_det) > 0 else np.zeros((1, 7), dtype=np.float32)

        ret = {
            'input': inp,
            'hm': hm,
            'reg_mask': reg_mask,
            'ind': ind,
            'wh': wh,
            'reg': reg,
            'gt_dets': gt_det,
            'gt_areas': gt_areas,
        }

        if self.num_keypoints > 0:
            ret['kps'] = kp
            ret['gt_kps'] = gt_kp
            ret['kp_reg_mask'] = kp_reg_mask
            del kpts_aug

        # sungting
        # if num_objs > 10:
        #     from sungting.get_det import get_detection

        #     res = get_detection(ret)
        #     bbox = res['gt_boxes'][0]

        #     if np.any((bbox[:, -1] > 30) & (bbox[:, -1] < 60)):
        #         radar_img = np.dstack((inp[0, ...],) * 3)
        #         np_img = np.interp(radar_img, (radar_img.min(), radar_img.max()), (0, 255)).astype(np.uint8)

        #         for b in bbox:
        #             cx, cy, width, height, theta = b
        #             minAreaRect = ((cx, cy), (width, height), theta)
        #             rectCnt = np.int64(cv2.boxPoints(minAreaRect))
        #             cv2.drawContours(np_img, [rectCnt], 0, (0,255,0), 3)
                
        #         from time import time
        #         time = int(time())
        #         cv2.imwrite('rotbbox{}.png'.format(time), np_img)
        #         cv2.imwrite('kp_img{}.png'.format(time), kp_img)

        return ret

    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox
