import logging

import torch
import numpy as np
from backends.decode import decode_detection
from utils.helper import CustomDataParallel, load_model, save_model

log = logging.getLogger(__name__)


class Model():
    def __init__(self):
        self.cfg = None
        self.backend = None
        self.optimizer = None
        self.centernet_loss = None
        self.device = None
        self.scheduler = None

        super().__init__()

    def init_done(self):
        pass

    def epoch_start(self):
        pass

    def epoch_end(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def step(self, data, is_training=True):
        for k in data:
            data[k] = data[k].to(device=self.device, non_blocking=True)

        if is_training:
            self.optimizer.zero_grad()

        outputs_source_domain = self.backend(data["input"], data, is_training)

        outputs = {
            "source_domain": outputs_source_domain
        }
        loss, stats = self.criterion(outputs, data)

        if is_training:
            if self.cfg.is_two_stage:
                loss += outputs_source_domain['two_stage_traning_loss']
            loss.backward()
            self.optimizer.step()

        stats["total_loss"] = loss

        for s in stats:
            try:
                stats[s] = stats[s].cpu().detach()
            except:
                import ipdb; ipdb.set_trace(context=7)

        outputs["stats"] = stats

        return outputs

    def set_phase(self, is_training=True):
        if is_training:
            self.backend.train()
        else:
            self.backend.eval()

    def to(self, device, parallel=False):
        if parallel:
            self.backend = CustomDataParallel(self.backend)

        self.backend.to(device)

    def criterion(self, outputs, batch):
        return self.centernet_loss(outputs["source_domain"], batch)

    def get_detections(self, outputs, batch):
        src = outputs["source_domain"]

        two_stage_score = src['testing_pred_cls'].reshape((self.cfg.batch_size, -1, 1))  
        two_stage_offset = src['testing_pred_offset'].reshape((self.cfg.batch_size, -1, 1))

        dets = decode_detection(
            src["hm"],
            src["wh"],
            src["reg"],
            kps=src["kps"] if 'kps' in src else None,
            K=self.cfg.max_detections,
            rotated=self.cfg.model.backend.params.rotated_boxes)
        
        # import ipdb; ipdb.set_trace(context=7)
        

        if 'kps' in src:
            dets, kps = dets
            kps[..., 0:2] *= self.backend.down_ratio
            kps = kps.detach().cpu().numpy()

        dets = dets.detach().cpu().numpy()
        dets[:, :, :4] *= self.backend.down_ratio

        ids = batch["id"].cpu().numpy()
        mask = (batch["reg_mask"].detach().cpu().numpy() == 1).squeeze()
        # sungting
        mask = mask.reshape(batch['reg_mask'].shape[0], -1)
        dets_gt = batch["gt_dets"].cpu().numpy()
        areas_gt = batch["gt_areas"].cpu().numpy()
        dets_gt[:, :, :4] *= self.backend.down_ratio

        if 'kps' in src:
            kps_gt = batch['gt_kps'].cpu().numpy() * self.backend.down_ratio

        gt_boxes = []
        gt_clss = []
        gt_ids = []
        gt_areas = []
        gt_kps = []

        box_idx = 4
        cls_idx = 5

        if self.cfg.model.backend.params.rotated_boxes:
            box_idx = 5
            cls_idx = 6

        for i in range(dets_gt.shape[0]):
            det_gt = dets_gt[i, mask[i]]

            gt_boxes.append(det_gt[:, :box_idx])
            gt_clss.append(det_gt[:, cls_idx].astype(np.int32))
            gt_ids.append(ids[i])
            gt_areas.append(areas_gt[i, mask[i]])

            if 'kps' in src:
                gt_kps.append(kps_gt[i, mask[i]])

        two_stage_score = dets[:, :, box_idx].copy()
        two_stage_offset = np.zeros_like(dets[:, :, :box_idx])

        cls = torch.sigmoid(src['testing_pred_cls']).reshape(len(src['pred_entry_coord']), -1).cpu().numpy()
        reg = src['testing_pred_offset'].reshape(len(src['pred_entry_coord']), -1, \
             src['testing_pred_offset'].shape[-1]).cpu().numpy()

        for i in range(two_stage_score.shape[0]):
            two_stage_score[i, :len(src['pred_entry_coord'][i])] = cls[i, :len(src['pred_entry_coord'][i])]
            two_stage_offset[i, :len(src['pred_entry_coord'][i])] = reg[i, :len(src['pred_entry_coord'][i])]

        def sigmoid(z):
            return 1/(1 + np.exp(-z))
        
        out = {
            # 'pred_boxes': dets[:, :, :box_idx],
            'pred_boxes': dets[:, :, :box_idx] + two_stage_offset,
            'pred_classes': dets[:, :, cls_idx].astype(np.int32),
            'pred_scores': np.sqrt(dets[:, :, box_idx] * two_stage_score),
            'gt_boxes': gt_boxes,
            'gt_classes': gt_clss,
            'gt_ids': gt_ids,
            'gt_areas': gt_areas,
            'two_stage_score': two_stage_score,
            'two_stage_offset': two_stage_offset
        }

        if 'kps' in src:
            out['gt_kps'] = gt_kps
            out['pred_kps'] = kps

        return out

    def load_model(self, path, resume=False):
        return load_model(self.backend, self.optimizer,
                          self.scheduler, path, resume)

    def save_model(self, path, epoch, with_optimizer=False):
        if with_optimizer:
            save_model(
                self.backend,
                path,
                epoch,
                self.optimizer,
                self.scheduler)
        else:
            save_model(
                self.backend,
                path,
                epoch)
