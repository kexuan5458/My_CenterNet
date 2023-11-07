from torch.utils.tensorboard import SummaryWriter
from utils.visualize import Visualizer
import os
import cv2


class TensorboardLogger:
    def __init__(self, cfg, classes):
        super().__init__()
        self.classes = classes
        self.summary_writer = SummaryWriter('logs')
        self.visualizer = Visualizer(
            classes,
            cfg.tensorboard.score_threshold,
            cfg.normalize.mean,
            cfg.normalize.std,
            font_size=cfg.tensorboard.font_size,
            alpha=cfg.tensorboard.alpha)
        self.num_visualizations = cfg.tensorboard.num_visualizations
        self.log_callback = None
        self.num_logged_images = 0
        self.cfg = cfg

    def log_detections(self, batch, detections, step, tag):
        # if self.num_logged_images >= self.num_visualizations:
        #     return

        images = batch["input"].detach().cpu().numpy()

        # sungting
        try:
            os.mkdir('npz_output')
        except:
            pass

        # sungting
        if self.cfg.save_npz:
            try:
                os.mkdir('npz_output')
            except:
                pass
            output_saver(detections, images.shape[0])

        ids = batch["id"].detach().cpu().numpy()
        for i in range(images.shape[0]):
            result, single_img = self.visualizer.visualize_detections(
                images[i].transpose(1, 2, 0),
                detections['pred_boxes'][i],
                detections['pred_classes'][i],
                detections['pred_scores'][i],
                detections['gt_boxes'][i],
                detections['gt_classes'][i],
                detections['gt_kps'][i] if 'gt_kps' in detections else None,
                detections['pred_kps'][i] if 'pred_kps' in detections else None,
            )

            # self.summary_writer.add_image(
            #     f'{tag}/detection_{ids[i]}', result, step)
            
            if self.cfg.is_vis_output:
                try:
                    # os.mkdir("result")
                    os.mkdir("single_result")
                except:
                    pass
                # cv2.imwrite("result/{:06d}.png".format(self.num_logged_images), result)
                cv2.imwrite("single_result/{:06d}.png".format(self.num_logged_images), single_img)

            self.num_logged_images += 1

            # if self.num_logged_images >= self.num_visualizations:
            #     break

    def log_stat(self, name, value, step):
        self.summary_writer.add_scalar(name, value, step)

    def log_image(self, name, image, step):
        self.summary_writer.add_image(name, image, step)

    def reset(self):
        self.num_logged_images = 0


import numpy as np
def output_saver(detections, batch_size):
    for i in range(batch_size):
        try:
            output_saver.counter += 1
        except AttributeError:
            output_saver.counter = 0
        np.savez('npz_output/{:06d}.npz'.format(output_saver.counter), pred_boxes=detections['pred_boxes'][i], pred_classes=detections['pred_classes'][i], 
                            pred_scores=detections['pred_scores'][i], gt_boxes=(detections['gt_boxes'][i]),
                            gt_classes=(detections['gt_classes'][i]))