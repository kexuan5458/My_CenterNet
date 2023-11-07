import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models.detection._utils as det_utils
import uda._mod_from_pytorch as mod_from_pytorch
from pycocotools import mask as maskUtils
import cv2
import dotsi


FG_IOU_THRESH = 0.5
BG_IOU_THRESH = 0.2

def get_gt(batch):
    ids = batch["id"].cpu().numpy()
    mask = (batch["reg_mask"].detach().cpu().numpy() == 1).squeeze()
    mask = mask.reshape(batch['reg_mask'].shape[0], -1)
    dets_gt = batch["gt_dets"].cpu().numpy()
    areas_gt = batch["gt_areas"].cpu().numpy()

    gt_boxes = []
    gt_clss = []
    gt_ids = []
    gt_areas = []
    gt_kps = []

    box_idx = 5
    cls_idx = 6

    for i in range(dets_gt.shape[0]):
        det_gt = dets_gt[i, mask[i]]

        gt_boxes.append(det_gt[:, :box_idx])
        gt_clss.append(det_gt[:, cls_idx].astype(np.int32))
        gt_ids.append(ids[i])
        gt_areas.append(areas_gt[i, mask[i]])

    for i in range(len(gt_boxes)):
        gt_boxes[i][:, :box_idx][..., -1] -= 90

    out_gt = {
        'gt_boxes': gt_boxes,
        'gt_classes': gt_clss,
        'gt_ids': gt_ids,
        'gt_areas': gt_areas
    }


    from utils.box import rotate_bbox

    batches_coord = []

    for i in (range(len(gt_boxes))): # per batch
        bbox_candidate = []
        for j in range(gt_boxes[i].shape[0]):
            cid = int(gt_clss[i][j])
            rot_pts = np.array(rotate_bbox(*gt_boxes[i][j]))
            contours = np.array(
                [rot_pts[0],
                rot_pts[1],
                rot_pts[2],
                rot_pts[3]])
            bbox_candidate.append(contours)
        batches_coord.append(bbox_candidate)

    return batches_coord, out_gt


def get_det(z):
    from backends.decode import decode_detection

    dets = decode_detection(
        torch.clamp(z["hm"].clone().detach().sigmoid_(), min=1e-4, max=1-1e-4),
        z["wh"].clone().detach(),
        z["reg"].clone().detach(),
        kps=z["kps"] if 'kps' in z else None,
        K=128,
        rotated=True)

    dets = dets.detach().cpu().numpy()
    box_idx = 5
    cls_idx = 6
    
    # sungting: shift angle to [-90, 90]
    dets[:, :, :box_idx][..., -1] -= 90

    out = {
        'pred_boxes': dets[:, :, :box_idx],
        'pred_classes': dets[:, :, cls_idx].astype(np.int32),
        'pred_scores': dets[:, :, box_idx]
    }

    

    pred_boxes = out['pred_boxes']
    pred_scores = out['pred_scores']
    pred_classes = out['pred_classes']

    from utils.box import rotate_bbox

    batches_out = []


    box_out = [] # list of (N, 5) x y w h ang

    for i in (range(pred_boxes.shape[0])):
        bbox_candidate = []
        ori_format_candidate = []
        for j in range(pred_boxes[i].shape[0]):
            if pred_scores[i][j] < 0.2:
                continue

            cid = int(pred_classes[i][j])
            rot_pts = np.array(rotate_bbox(*pred_boxes[i][j]))
            contours = np.array(
                [rot_pts[0],
                rot_pts[1],
                rot_pts[2],
                rot_pts[3]])
            
            candidate = {
                'corner_pts': contours,
                'scores': pred_scores[i][j],
                'pred_cls': cid + 1,
                'entry_coord': (i, j)
            }

            bbox_candidate.append(candidate)
            ori_format_candidate.append(pred_boxes[i][j])

        box_out.append(np.array(ori_format_candidate))
        batches_out.append(bbox_candidate)
    
    return batches_out, box_out
        
def bilinear_interpolate_torch(feature_map, x, y):
    """
    Args:
        im: (H, W, C) [y, x]
        x: (N)
        y: (N)
    Returns:
    """
    im = feature_map.permute((1, 2, 0)).contiguous()

    x = x.to(device='cuda')
    y = y.to(device='cuda')

    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)
    
    return ans

def assign_targets_to_proposals(gt_boxes, proposals):
    gt_labels = [torch.ones(len(x)) for x in gt_boxes]
    # originally: proposals, gt_boxes, gt_labels: [batch1, batch2...]
    fg_iou_thresh = FG_IOU_THRESH
    bg_iou_thresh = BG_IOU_THRESH

    proposal_matcher = det_utils.Matcher(
        fg_iou_thresh,
        bg_iou_thresh,
        allow_low_quality_matches=False)

    matched_idxs = []
    labels = []
    all_iou_matrix = []

    for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):
        # proposals_in_image, gt_boxes_in_image, gt_labels_in_image = torch.tensor(proposals_in_image), torch.tensor(gt_boxes_in_image), torch.tensor(gt_labels_in_image)
        if len(gt_boxes_in_image) == 0:
            # Background image
            device = 'cuda'
            clamped_matched_idxs_in_image = torch.zeros(
                (len(proposals_in_image),), dtype=torch.int64, device=device
            )
            labels_in_image = torch.zeros(
                (len(proposals_in_image),), dtype=torch.int64, device=device
            )
            all_iou_matrix.append(torch.tensor([], dtype=torch.int64, device=device
            ))
            # import ipdb; ipdb.set_trace(context=7)
            
        else: # has gt, has no pred will also go to this
            #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
            # TODO

            # shape: (gt_num, proposal_num)
            # match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)

            # shape: (pred_num, gt_num) => (gt_num, proposal_num)
            # import ipdb; ipdb.set_trace(context=7)

            if len(proposals_in_image) == 0:
                device = 'cuda'

                clamped_matched_idxs_in_image = torch.tensor([], dtype=torch.int64, device=device)
                labels_in_image = torch.tensor([], dtype=torch.int64, device=device)

                all_iou_matrix.append(torch.tensor([], dtype=torch.int64, device=device))
                matched_idxs.append(clamped_matched_idxs_in_image)
                labels.append(labels_in_image)

                continue
            
            try:
                match_quality_matrix = np.array(computeIoU(gt_boxes_in_image, proposals_in_image)).T
            except:
                import ipdb; ipdb.set_trace(context=7)
                
            match_quality_matrix = torch.tensor(match_quality_matrix)
            all_iou_matrix.append(match_quality_matrix)

            matched_idxs_in_image = proposal_matcher(match_quality_matrix)

            clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

            labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
            labels_in_image = labels_in_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs_in_image == proposal_matcher.BELOW_LOW_THRESHOLD
            labels_in_image[bg_inds] = 0

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs_in_image == proposal_matcher.BETWEEN_THRESHOLDS
            labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

        matched_idxs.append(clamped_matched_idxs_in_image)
        labels.append(labels_in_image)
    return matched_idxs, labels, all_iou_matrix

def box_to_seg(bb_pts):
    batch_size = len(bb_pts)
    batch_out = []
    image_shape = (288, 288)

    for i in range(batch_size):
        this_batch = []
        for bb in bb_pts[i]:
            mask = np.zeros((image_shape))
            rot_pts = np.array(bb)
            cv2.fillPoly(mask, [rot_pts.reshape(1, -1, 2)], color=(1,))
            mask = np.asfortranarray(mask.astype(np.uint8))
            rle = maskUtils.encode(mask)
            ar = maskUtils.area(rle)

            out = dict(
                segmentation = rle,
                area = ar,
                iscrowd = 0
            )

            this_batch.append(out)

        batch_out.append(this_batch)
    
    return batch_out

def computeIoU(gt, dt):
    # iouType == 'segm':
    g = [g['segmentation'] for g in gt]
    d = [d['segmentation'] for d in dt]

    iscrowd = [int(o['iscrowd']) for o in gt]
    ious = maskUtils.iou(d,g,iscrowd)

    return ious

def dump():
    from pycocotools import mask as maskUtils
    import cv2

    gt = np.zeros((288, 288), dtype=np.uint8)
    for s in gt_segs[0]:
        seg = s['segmentation']
        m = maskUtils.decode(seg)
        gt = (gt | m)

    pred = np.zeros((288, 288), dtype=np.uint8)
    for s in pred_segs[0]:
        seg = s['segmentation']
        m = maskUtils.decode(seg)
        pred = (pred | m)

    final = np.dstack((np.zeros_like(gt), pred*255, gt*255))
    cv2.imwrite('3mask.png', final)


def subsample(labels):
    batch_size_per_image = 128
    positive_fraction = 0.25

    fg_bg_sampler = mod_from_pytorch.BalancedPositiveNegativeSampler(
        batch_size_per_image,
        positive_fraction)

    sampled_pos_inds, sampled_neg_inds = fg_bg_sampler(labels)
    sampled_inds = []

    for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
        zip(sampled_pos_inds, sampled_neg_inds)
    ):
        img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
        sampled_inds.append(img_sampled_inds)
    return sampled_inds

INPUT_CHANNELS = 326
SHARED_FC = (256, 256)
DP_RATIO = 0.3
CLS_FC = [256, 256]
REG_FC = [256, 256]
NUM_CLASS = 1
CODE_SIZE = 5

# TODO
class TwoStageNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.build_network()

        self.model_cfg = dotsi.Dict(dict(
            CLASS_AGNOSTIC=True,
            SHARED_FC=[256, 256],
            CLS_FC=[256, 256],
            REG_FC=[256, 256],
            DP_RATIO=0.3,

            TARGET_CONFIG=dict(
                ROI_PER_IMAGE=128,
                FG_RATIO=0.5,
                SAMPLE_ROI_BY_EACH_CLASS=True,
                CLS_SCORE_TYPE='roi_iou',
                CLS_FG_THRESH=0.75,
                CLS_BG_THRESH=0.25,
                CLS_BG_THRESH_LO=0.1,
                HARD_BG_RATIO=0.8,
                REG_FG_THRESH=0.55
            ),
            LOSS_CONFIG=dict(
                # CLS_LOSS='BinaryCrossEntropy',
                CLS_LOSS='FocalLoss',
                REG_LOSS='L1',
                LOSS_WEIGHTS={
                    'rcnn_cls_weight': 1.0,
                    'rcnn_reg_weight': 1.0,
                    'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0]
                }
            )
        ))

    def build_network(self):
        pre_channel = INPUT_CHANNELS

        shared_fc_list = []
        for k in range(0, SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Conv1d(pre_channel, SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = SHARED_FC[k]

            if k != SHARED_FC.__len__() - 1 and DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(DP_RATIO))

        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        self.cls_layers = self.make_fc_layers(
            input_channels=pre_channel, output_channels=NUM_CLASS, fc_list=CLS_FC
        )

        self.reg_layers = self.make_fc_layers(
            input_channels=pre_channel,
            output_channels=CODE_SIZE,
            fc_list=REG_FC
        )

    def make_fc_layers(self, input_channels, output_channels, fc_list):
        fc_layers = []
        pre_channel = input_channels
        for k in range(0, fc_list.__len__()):
            fc_layers.extend([
                nn.Conv1d(pre_channel, fc_list[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(fc_list[k]),
                nn.ReLU()
            ])
            pre_channel = fc_list[k]
            if DP_RATIO >= 0 and k == 0:
                fc_layers.append(nn.Dropout(DP_RATIO))
        fc_layers.append(nn.Conv1d(pre_channel, output_channels, kernel_size=1, bias=True))
        fc_layers = nn.Sequential(*fc_layers)
        return fc_layers
    
    def forward(self, two_stage_batch, z_dict, is_training):
        pooled_features = two_stage_batch['roi_feature'].reshape(-1, 1,
            two_stage_batch['roi_feature'].shape[-1]).contiguous()  # (BxN, 1, C)
        batch_size_rcnn = pooled_features.shape[0]
        pooled_features = pooled_features.permute(0, 2, 1).float().contiguous()

        shared_features = self.shared_fc_layer(pooled_features.view(batch_size_rcnn, -1, 1))

        rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)

        two_stage_batch['pred_cls'] = rcnn_cls
        two_stage_batch['pred_offset'] = rcnn_reg

        if is_training:
            self.forward_ret_dict = two_stage_batch
        else:
            z_dict['testing_pred_cls'] = rcnn_cls
            z_dict['testing_pred_offset'] = rcnn_reg

        # return two_stage_batch

    def get_box_reg_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        code_size = forward_ret_dict['pred_offset'].shape[-1]
        reg_valid_mask = forward_ret_dict['reg_valid_mask'].view(-1)
        gt_boxes3d_ct = forward_ret_dict['gt_roi_offset'][..., 0:code_size]
        rcnn_reg = forward_ret_dict['pred_offset']  # (rcnn_batch_size, C)
        rcnn_batch_size = gt_boxes3d_ct.view(-1, code_size).shape[0]

        fg_mask = (reg_valid_mask > 0)
        fg_sum = fg_mask.long().sum().item()

        tb_dict = {}
        
        if loss_cfgs.REG_LOSS == 'L1':
            reg_targets = gt_boxes3d_ct.view(rcnn_batch_size, -1)
            rcnn_loss_reg = F.l1_loss(
                rcnn_reg.view(rcnn_batch_size, -1),
                reg_targets,
                reduction='none'
            )  # [B, M, 5]

            rcnn_loss_reg = rcnn_loss_reg * rcnn_loss_reg.new_tensor(\
                loss_cfgs.LOSS_WEIGHTS['code_weights'])

            rcnn_loss_reg = (rcnn_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(fg_sum, 1)
            rcnn_loss_reg = rcnn_loss_reg * loss_cfgs.LOSS_WEIGHTS['rcnn_reg_weight']
            tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.detach()
        else:
            raise NotImplementedError

        return rcnn_loss_reg, tb_dict

    def get_box_cls_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        rcnn_cls = forward_ret_dict['pred_cls']
        rcnn_cls_labels = forward_ret_dict['gt_cls'].view(-1)


        if loss_cfgs.CLS_LOSS == 'BinaryCrossEntropy':
            rcnn_cls_flat = rcnn_cls.view(-1)
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), rcnn_cls_labels.float(), reduction='none')
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)

        if loss_cfgs.CLS_LOSS == 'FocalLoss':
            focal_loss = FocalLoss(weight=1.0)

            rcnn_cls_flat = rcnn_cls.view(-1)
            batch_loss_cls = focal_loss(torch.sigmoid(rcnn_cls_flat), rcnn_cls_labels.float())

            rcnn_loss_cls = batch_loss_cls

        elif loss_cfgs.CLS_LOSS == 'CrossEntropy':
            batch_loss_cls = F.cross_entropy(rcnn_cls, rcnn_cls_labels, reduction='none', ignore_index=-1)
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        else:
            raise NotImplementedError

        rcnn_loss_cls = rcnn_loss_cls * loss_cfgs.LOSS_WEIGHTS['rcnn_cls_weight']
        tb_dict = {'rcnn_loss_cls': rcnn_loss_cls.detach()}
        return rcnn_loss_cls, tb_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        rcnn_loss = 0
        rcnn_loss_cls, cls_tb_dict = self.get_box_cls_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_cls
        tb_dict.update(cls_tb_dict)

        rcnn_loss_reg, reg_tb_dict = self.get_box_reg_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_reg
        tb_dict.update(reg_tb_dict)
        tb_dict['rcnn_loss'] = rcnn_loss
        return rcnn_loss, tb_dict

class FocalLoss(torch.nn.Module):
    '''nn.Module warpper for focal loss'''

    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, out, target):
        return self._neg_loss(out, target)

    def _neg_loss(self, pred, gt):
        ''' Modified focal loss. Exactly the same as CornerNet.
            Runs faster and costs a little bit more memory
            Arguments:
            pred (batch x c x h x w)
            gt_regr (batch x c x h x w)
        '''
        pos_inds = gt.ge(0.7).float()
        neg_inds = gt.lt(0.7).float()

        neg_weights = torch.pow(1 - gt, 4)

        loss = 0

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * \
            neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss * self.weight