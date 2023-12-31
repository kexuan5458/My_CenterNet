import numpy as np
from matplotlib.pyplot import get_cmap
from imgaug.augmentables import BoundingBox, Polygon, Keypoint
from utils.box import rotate_bbox
import cv2

# cm = get_cmap('gist_rainbow')
cm = get_cmap('Set2')


class Visualizer:
    def __init__(self, classes, score_threshold,
                 mean, std, font_size=14, alpha=0.5):
        self.classes = classes
        self.score_threshold = score_threshold
        self.font_size = font_size
        self.mean = mean
        self.std = std
        self.alpha = alpha
        self.cmap = [
            [int(y * 255.0) for y in cm(1.0 * x / len(self.classes))[: 3]]
            for x in range(len(self.classes))]
        # self.cmap = [(x[1], x[2], x[0]) for x in self.cmap]
        self.cmap = [x[::-1] for x in self.cmap]

    def visualize_detections(
            self, image, pred_boxes, pred_classes, pred_scores, gt_boxes,
            gt_classes, gt_kps=None, pred_kps=None):

        pred_img = (
            (np.dstack((image[..., 0],)*3) *
             self.std +
             self.mean) *
            255).astype(
            np.uint8).copy()
        gt_img = pred_img.copy()

        if gt_boxes.shape[-1] == 5:
            pred_img, gt_img, single_img = self.__draw_rotated_boxes(
                gt_img, gt_boxes, gt_classes, pred_img, pred_boxes,
                pred_scores, pred_classes)
        else:
            pred_img, gt_img, single_img = self.__draw_boxes(
                gt_img, gt_boxes, gt_classes, pred_img, pred_boxes,
                pred_scores, pred_classes)

        if pred_kps is not None:
            pred_img, gt_img = self.__draw_keypoints(
                gt_img, gt_kps, pred_img, pred_kps, pred_scores)

        result = np.hstack([pred_img, gt_img])
        # return result.transpose(2, 0, 1)

        return result, single_img

    def __draw_boxes(self, gt_img, gt_boxes, gt_classes,
                     pred_img, pred_boxes, pred_scores, pred_classes):
        for i in range(gt_boxes.shape[0]):
            cid = int(gt_classes[i])
            bb = BoundingBox(*gt_boxes[i])
            bb.label = self.classes[cid]['name']
            gt_img = bb.draw_box_on_image(
                gt_img, self.cmap[cid], self.alpha, 3)
            gt_img = bb.draw_label_on_image(
                gt_img,
                self.cmap[cid],
                size_text=self.font_size,
                alpha=self.alpha,
                height=self.font_size + 4)
        
        single_img = gt_img.copy()

        for i in range(pred_boxes.shape[0]):
            if pred_scores[i] < self.score_threshold:
                continue

            cid = int(pred_classes[i])
            bb = BoundingBox(*pred_boxes[i])
            # bb.label = f"{self.classes[int(cid)]['name']}: {pred_scores[i]:.2f}"
            bb.label = f"{pred_scores[i]:.2f}"

            pred_img = bb.draw_box_on_image(
                pred_img, self.cmap[cid], self.alpha, 3)
            pred_img = bb.draw_label_on_image(
                pred_img,
                self.cmap[cid],
                size_text=self.font_size,
                alpha=self.alpha,
                height=self.font_size + 4)

            single_img = bb.draw_box_on_image(
                single_img, self.cmap[cid], self.alpha, 3)
            single_img = bb.draw_label_on_image(
                single_img,
                self.cmap[cid],
                size_text=self.font_size,
                alpha=self.alpha,
                height=self.font_size + 4)

        return pred_img, gt_img, single_img

    def __draw_rotated_boxes(self, gt_img, gt_boxes, gt_classes,
                             pred_img, pred_boxes, pred_scores, pred_classes):
        

        # sungting
        gt_boxes[:, -1] -= 90
        pred_boxes[:, -1] -= 90

        for i in range(gt_boxes.shape[0]):
            cid = int(gt_classes[i])
            rot_pts = np.array(rotate_bbox(*gt_boxes[i]))
            contours = np.array(
                [rot_pts[0],
                 rot_pts[1],
                 rot_pts[2],
                 rot_pts[3]])
            # bb.label = self.classes[cid]['name']
            cv2.polylines(
                gt_img,
                [contours],
                isClosed=True,
                # color=self.cmap[cid],
                color=(255, 255, 255),
                thickness=2,
                lineType=cv2.LINE_AA)

            # poly = Polygon(
            #     np.array(contours, dtype=np.int32).reshape(-1, 2),
            #     label=self.classes[cid]['name'])

            # gt_img = poly.to_bounding_box().draw_label_on_image(
            #     gt_img,
            #     self.cmap[cid],
            #     size_text=self.font_size,
            #     alpha=self.alpha,
            #     height=self.font_size + 4)
        
        single_img = gt_img.copy()

        for i in range(pred_boxes.shape[0]):
            if pred_scores[i] < self.score_threshold:
                continue

            cid = int(pred_classes[i])
            rot_pts = np.array(rotate_bbox(*pred_boxes[i]))
            contours = np.array(
                [rot_pts[0],
                 rot_pts[1],
                 rot_pts[2],
                 rot_pts[3]])

            # bb.label = f"{self.classes[int(cid)]['name']}: {pred_scores[i]:.2f}"
            poly = Polygon(
                np.array(contours, dtype=np.int32).reshape(-1, 2),
                label=f"{pred_scores[i]:.2f}")
                # label=f"{self.classes[int(cid)]['name']}: {pred_scores[i]:.2f}")
            # pred_img = poly.draw_on_image(pred_img, self.cmap[cid], alpha=self.alpha, )

            cv2.polylines(
                pred_img,
                [contours],
                isClosed=True,
                color=self.cmap[cid],
                thickness=1,
                lineType=cv2.LINE_AA)

            pred_img = poly.to_bounding_box().draw_label_on_image(
                pred_img,
                self.cmap[cid],
                size_text=self.font_size,
                alpha=self.alpha,
                height=self.font_size + 4)

            cv2.polylines(
                single_img,
                [contours],
                isClosed=True,
                color=self.cmap[cid],
                thickness=1,
                lineType=cv2.LINE_AA)

            single_img = poly.to_bounding_box().draw_label_on_image(
                single_img,
                self.cmap[cid],
                size_text=self.font_size,
                alpha=self.alpha,
                height=self.font_size + 4)

        return pred_img, gt_img, single_img

    def __draw_keypoints(self, gt_img, gt_kps,
                         pred_img, pred_kps, pred_scores):
        for i in range(gt_kps.shape[0]):
            for p in gt_kps[i]:
                kp = Keypoint(*p)
                gt_img = kp.draw_on_image(
                    gt_img, (0, 255, 255), self.alpha, 3)

        for i in range(pred_kps.shape[0]):
            if pred_scores[i] < self.score_threshold:
                continue

            for p in pred_kps[i]:
                kp = Keypoint(*p)
                pred_img = kp.draw_on_image(
                    pred_img, (0, 255, 255), self.alpha, 3)
        return pred_img, gt_img
