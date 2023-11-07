from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging
import numpy as np
from os.path import join

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchsummary import summary

from libs.DCNv2.dcn_v2 import DCN
from uda.two_stage import BG_IOU_THRESH, FG_IOU_THRESH

import cv2
import matplotlib.pyplot as plt

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):
    return join('http://dl.yf.io/dla/models', data,
                '{}-{}.pth'.format(name, hash))


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class BottleneckX(nn.Module):
    expansion = 2
    cardinality = 32

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BottleneckX, self).__init__()
        cardinality = BottleneckX.cardinality
        # dim = int(math.floor(planes * (BottleneckV5.expansion / 64.0)))
        # bottle_planes = dim * cardinality
        bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation, bias=False,
                               dilation=dilation, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000,
                 block=BasicBlock, residual_root=False, linear_root=False):
        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            # nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
            nn.Conv2d(5, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False,
                           root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        return y

    def load_pretrained_model(self, data='imagenet',
                              name='dla34', hash='ba72cf86'):
        # fc = self.fc
        if name.endswith('.pth'):
            model_weights = torch.load(data + name)
        else:
            model_url = get_model_url(data, name, hash)
            model_weights = model_zoo.load_url(model_url)
        num_classes = len(model_weights[list(model_weights.keys())[-1]])
        self.fc = nn.Conv2d(
            self.channels[-1], num_classes,
            kernel_size=1, stride=1, padding=0, bias=True)
        self.load_state_dict(model_weights)
        # self.fc = fc


def dla34(pretrained=True, **kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock, **kwargs)
    if pretrained:
        model.load_pretrained_model(
            data='imagenet', name='dla34', hash='ba72cf86')
    return model


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class DeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.conv = DCN(
            chi,
            cho,
            kernel_size=(
                3,
                3),
            stride=1,
            padding=1,
            dilation=1,
            deformable_groups=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x


class IDAUp(nn.Module):

    def __init__(self, o, channels, up_f):
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])
            proj = DeformConv(c, o)
            node = DeformConv(o, o)

            up = nn.ConvTranspose2d(o, o, f * 2, stride=f,
                                    padding=f // 2, output_padding=0,
                                    groups=o, bias=False)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)

    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])


class DLAUp(nn.Module):
    def __init__(self, startp, channels, scales, in_channels=None):
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(channels[j], in_channels[j:],
                          scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]]  # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) - i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


class Interpolate(nn.Module):
    def __init__(self, scale, mode):
        super(Interpolate, self).__init__()
        self.scale = scale
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(
            x,
            scale_factor=self.scale,
            mode=self.mode,
            align_corners=False)
        return x


class DLASeg(nn.Module):
    def __init__(
            self, base_name, heads, pretrained, down_ratio, final_kernel,
            last_level, head_conv, out_channel=0, freeze_base=False,
            rotated_boxes=False,
            cfg=None):
        super(DLASeg, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.down_ratio = down_ratio
        self.rotated_boxes = rotated_boxes
        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level
        self.base = globals()[base_name](pretrained=pretrained)
        self.cfg = cfg

        if freeze_base:
            for layer in self.base.parameters():
                layer.requires_grad = False

        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(
            self.first_level, channels[self.first_level:],
            scales)

        if out_channel == 0:
            out_channel = channels[self.first_level]

        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level], [
                            2 ** i for i in range(self.last_level - self.first_level)])

        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(channels[self.first_level], head_conv,
                              kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes,
                              kernel_size=final_kernel, stride=1,
                              padding=final_kernel // 2, bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(channels[self.first_level], classes,
                               kernel_size=final_kernel, stride=1,
                               padding=final_kernel // 2, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

        from uda.two_stage import TwoStageNet
        self.roi_head = TwoStageNet()

    def forward(self, x, batch, is_training):
        # import ipdb; ipdb.set_trace(context=7)
        
        x = self.base(x)
        x = self.dla_up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))

        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(y[-1])

        # save_featuremap(z['hm'])

        # sungting start
        """
        use first stage's result as anchor to generate proposals, and label as hit or not
        based on gt
        """
        from uda.two_stage import (get_det, bilinear_interpolate_torch,
                                   get_gt, box_to_seg, computeIoU,
                                   assign_targets_to_proposals, subsample,
                                   BG_IOU_THRESH, FG_IOU_THRESH)


        pred_bbox, pred_out = get_det(z)
        pred_segs = box_to_seg([[x['corner_pts'] for x in batch] for batch in pred_bbox])
        entry_coord = [[x['entry_coord'] for x in batch] for batch in pred_bbox]
        
        # TODO: change to only use gt when training
        if is_training: # when training
            gt_bbox, gt_out = get_gt(batch) # bbox coord: shape: (B, obj_num, 4, 2)
            gt_segs = box_to_seg(gt_bbox)

            # len of the following two is equal of the num of pred of first stage
            # import ipdb; ipdb.set_trace(context=7)
            
            # TODO: check no gt, no pred edge case
            # import ipdb; ipdb.set_trace(context=7)
            
            matched_idxs, labels, batch_iou_matrix = assign_targets_to_proposals(gt_segs, pred_segs) # batch_iou_matrix: shape=(gt, pred)
            pred_max_iou = [b.max(dim=0)[0] if len(b) > 0 else torch.tensor([]) for b in batch_iou_matrix]
            # pred_max_iou = [b.max(dim=0)[0] for b in batch_iou_matrix]
            
            labels = [label.to(torch.float64) for label in labels]
            sampled_inds = subsample(labels)

            ### fix intermediate batch_cls_labels value, sampled_inds: shape will be (batch, 128) afterward
            def fix_label(labels):
                for i in range(len(labels)):
                    intermed_selection = (labels[i] == -1)
                    iou = batch_iou_matrix[i]

                    if intermed_selection.sum() > 0:
                        # make "in between" pred has hit or not value between 0 and 1
                        val, ind = torch.max(iou[:, intermed_selection], 0)
                            
                        val = (val - BG_IOU_THRESH) / (FG_IOU_THRESH - BG_IOU_THRESH)
                        labels[i][intermed_selection] = val

                    total_sample = 128

                    positive_mask = (labels[i] == 1)
                    positive_num = positive_mask.sum()

                    negative_mask = (labels[i] == 0)
                    negative_num = negative_mask.sum()

                    # import ipdb; ipdb.set_trace(context=7)
                    # print(">>>> negative_num: ", negative_num)

                    between_mask = (negative_mask == 0) & (positive_mask == 0)
                    between_num = between_mask.sum()

                    all_left = (total_sample - positive_num - negative_num - between_num)

                    neg_left = all_left // 2
                    neg_left, _ = torch.max(neg_left, 0)

                    between_left = all_left - neg_left 
                    between_left, _ = torch.max(between_left, 0)

                    if negative_num == 0 and between_num == 0:
                        whole_sample = torch.zeros((128,), dtype=torch.long).to('cuda')
                        whole_sample[:len(sampled_inds[i])] = sampled_inds[i].to('cuda')

                    elif negative_num == 0:
                        between_left = all_left

                        between_idx = torch.arange(len(labels[i]))[between_mask]
                        between_idx_sample = between_idx[torch.randint(0, between_num, (between_left,))]

                        whole_sample = torch.cat((sampled_inds[i].to('cuda'), between_idx_sample.to('cuda')))
                    
                    elif between_num == 0:
                        neg_left = all_left

                        neg_idx = torch.arange(len(labels[i]))[negative_mask].to('cuda')
                        
                        try:
                            neg_idx_sample = neg_idx[torch.randint(0, negative_num, (neg_left,))]
                        except:
                            import ipdb; ipdb.set_trace(context=7)
                            

                        whole_sample = torch.cat((sampled_inds[i].to('cuda'), neg_idx_sample.to('cuda')))
                    
                    else:
                        neg_idx = torch.arange(len(labels[i]))[negative_mask]
                        neg_idx_sample = neg_idx[torch.randint(0, negative_num, (neg_left,))]
                        
                        between_idx = torch.arange(len(labels[i]))[between_mask]
                        between_idx_sample = between_idx[torch.randint(0, between_num, (between_left,))]

                        whole_sample = torch.cat((sampled_inds[i].to('cuda'), neg_idx_sample.to('cuda'), between_idx_sample.to('cuda')))

                    sampled_inds[i] = whole_sample

            # if (any(map(len, pred_out)) == True) and (any(map(len, gt_bbox)) == True):
            #     import ipdb; ipdb.set_trace(context=7)
            
            fix_label(labels)
            ### end

        features = []

        for i, batch_pred_box in enumerate([[x['corner_pts'] for x in batch] for batch in pred_bbox]):
            if len(batch_pred_box) == 0:
                features.append(torch.tensor([]))
                continue

            pt5 = get_box_center_4pt(batch_pred_box)
            pb = np.vstack(pt5)
            
            feature = bilinear_interpolate_torch(y[-1][i, ...], torch.tensor(pb[:, 0]), torch.tensor(pb[:, 1]))
            # feature = feature.reshape(-1, 5, feature.shape[-1])
            feature = feature.reshape(feature.shape[0]//5, -1)
            features.append(feature)

        scores = [torch.tensor([x['scores'] for x in batch]).to('cuda') for batch in pred_bbox]
        boxes = [torch.tensor([x['corner_pts'] for x in batch]).to('cuda') for batch in pred_bbox]
        pred_cls = [torch.tensor([x['pred_cls'] for x in batch]).to('cuda') for batch in pred_bbox]

        # self.cfg.test_only
        if is_training: # when training
            # merge roi_feature, roi_box, roi_score
            def get_merged_vector_training(matched_idxs, labels, sampled_inds, features):
                out = []

                roi_feature = torch.zeros((len(sampled_inds), 128, 326)).to('cuda')
                gt_cls = torch.zeros((len(sampled_inds), 128)).to('cuda')
                gt_roi_offset = torch.zeros((len(sampled_inds), 128, 5)).to('cuda')
                gt_init_box = torch.zeros((len(sampled_inds), 128, 5)).to('cuda')
                reg_valid_mask = torch.zeros((len(sampled_inds), 128)).to('cuda')

                for i in range(len(sampled_inds)): # iterate batch
                    if len(features[i]) == 0:
                        continue

                    feat = features[i][sampled_inds[i]]

                    roi_box = torch.tensor(pred_out[i])[sampled_inds[i]].to('cuda')
                    roi_box = roi_box.reshape(roi_box.shape[0], -1)

                    score = scores[i][sampled_inds[i]]
                    score = score.reshape(score.shape[0], -1)

                    final_feature = torch.cat((feat, roi_box, score), dim=-1)
                    batch_label = labels[i][sampled_inds[i]].to('cuda')

                    roi_feature[i, :final_feature.shape[0]] = final_feature
                    gt_cls[i, :batch_label.shape[0]] = batch_label

                    if len(gt_out['gt_boxes'][i]) == 0:
                        continue

                    matched_gt_id = matched_idxs[i][sampled_inds[i]]

                    try:
                        if len(pred_max_iou[i]) != 0:
                            valid_reg_mask = ((pred_max_iou[i][sampled_inds[i]]) > 0.5).to('cuda') # REG_FG_THRESH
                    except:
                        import ipdb; ipdb.set_trace(context=7)

                    try:
                        gt_box = torch.tensor(gt_out['gt_boxes'][i])[matched_gt_id].to('cuda')
                    except:
                        import ipdb; ipdb.set_trace(context=7)
                        
                    gt_box = gt_box.reshape(gt_box.shape[0], -1)


                    try:
                        gt_roi_offset[i, :gt_box.shape[0]] = gt_box - roi_box
                    except:
                        import ipdb; ipdb.set_trace(context=7)
                        
                    reg_valid_mask[i, :valid_reg_mask.shape[0]] = valid_reg_mask


                    # out.append({
                    #     'roi_feature': final_feature,
                    #     'gt_cls': batch_label,
                    #     'gt_roi_offset': gt_box - roi_box,
                    #     'reg_valid_mask': valid_reg_mask
                    # })

                    # out.append({
                    #     'roi_feature': roi_feature[i],
                    #     'gt_cls': gt_cls[i],
                    #     'gt_roi_offset': gt_roi_offset[i],
                    #     'reg_valid_mask': reg_valid_mask[i]
                    # })
                
                # batch_out = {
                #     'roi_feature': torch.cat([x['roi_feature'].unsqueeze(0) for x in out], dim=0),
                #     'gt_cls': torch.cat([x['gt_cls'].unsqueeze(0) for x in out], dim=0),
                #     'gt_roi_offset': torch.cat([x['gt_roi_offset'].unsqueeze(0) for x in out], dim=0),
                #     'reg_valid_mask': torch.cat([x['reg_valid_mask'].unsqueeze(0) for x in out], dim=0)
                # }
                
                batch_out = {
                    'roi_feature': roi_feature,
                    'gt_cls': gt_cls,
                    'gt_roi_offset': gt_roi_offset,
                    'reg_valid_mask': reg_valid_mask
                }
                return batch_out

            two_stage_batch = get_merged_vector_training(matched_idxs, labels, sampled_inds, features)
        else: # when inferencing
            # TODO
            def get_merged_vector_testing(features):
                out = []

                batch_size = len(features)

                feat = features[0].new_zeros((batch_size, 128, 320)).to('cuda')
                roi_box = torch.tensor(pred_out[0]).new_zeros((batch_size, 128, 5)).to('cuda')
                score = scores[0].new_zeros((batch_size, 128)).to('cuda')
                label_pred = pred_cls[0].new_zeros((batch_size, 128)).to('cuda')
                roi_feature = features[0].new_zeros((batch_size, 128, 326)).to('cuda')
                

                for i in range(batch_size): # iterate batch
                    num_obj = features[i].shape[0]

                    if len(features[i]) == 0:
                        continue

                    # feat = features[i][sampled_inds[i]]
                    feat[i, :num_obj] = features[i]
                        

                    # valid_reg_mask = ((pred_max_iou[i][sampled_inds[i]]) > 0.5).to('cuda') # REG_FG_THRESH

                    # roi_box = torch.tensor(pred_out[i][sampled_inds[i]]).to('cuda')
                    # roi_box = roi_box.reshape(roi_box.shape[0], -1)

                    roi_box[i, :num_obj] = torch.tensor(pred_out[i])
                        

                    # score = scores[i][sampled_inds[i]]
                    # score = score.reshape(score.shape[0], -1)

                    score[i, :num_obj] = scores[i]

                    final_feature = torch.cat((feat[i], roi_box[i], score[i].unsqueeze(-1)), dim=-1)
                    roi_feature[i] = final_feature


                    # batch_label = labels[i][sampled_inds[i]].to('cuda')
                    label_pred[i, :num_obj] = pred_cls[i]

                    # matched_gt_id = matched_idxs[i][sampled_inds[i]]
                    # gt_box = torch.tensor(gt_out['gt_boxes'][i][matched_gt_id]).to('cuda')
                    # gt_box = gt_box.reshape(gt_box.shape[0], -1)

                    # out.append({
                    #     'roi_feature': roi_feature,
                    #     'label_pred': label_pred[i]
                    #     # 'gt_cls': batch_label,
                    #     # 'gt_roi_offset': gt_box - roi_box,
                    #     # 'reg_valid_mask': valid_reg_mask
                    # })
                
                batch_out = {
                    # 'roi_feature': torch.cat([x['roi_feature'].unsqueeze(0) for x in out], dim=0),
                    'roi_feature': roi_feature,
                    'label_pred': label_pred
                    # 'gt_cls': torch.cat([x['gt_cls'].unsqueeze(0) for x in out], dim=0),
                    # 'gt_roi_offset': torch.cat([x['gt_roi_offset'].unsqueeze(0) for x in out], dim=0),
                    # 'reg_valid_mask': torch.cat([x['reg_valid_mask'].unsqueeze(0) for x in out], dim=0)
                }

                
                return batch_out

            two_stage_batch = get_merged_vector_testing(features)


        try:
            self.roi_head(two_stage_batch, z, is_training)
        except:
            import ipdb; ipdb.set_trace(context=7)
            

        if is_training: # training
            roi_loss, tb_dict = self.roi_head.get_loss()
            z['two_stage_traning_loss'] = roi_loss
            z['two_stage_traning_loss_dict'] = tb_dict
        
        else: # testing
            z['pred_entry_coord'] = entry_coord

        # end

        return z

def get_box_center_4pt(batch_pred_box):
    all_np_box = np.array(batch_pred_box)

    if len(all_np_box) == 0:
        return np.array([])
    else:
        four_edge_center = (all_np_box + np.roll(all_np_box, -1, axis=-2)) / 2
        center = ((all_np_box[:, 0, :] + all_np_box[:, 1, :]) / 2 + (all_np_box[:, 2, :] + all_np_box[:, 3, :]) / 2) / 2
        center_5 = np.concatenate((four_edge_center, center[:, None, :]), axis=-2)
        
    return center_5

def build(num_classes, num_keypoints=0, head_conv=256,
          down_ratio=4, freeze_base=False, rotated_boxes=False, cfg=None):

    heads = {
        'hm': num_classes,
        'wh': 2 if not rotated_boxes else 3,
        'reg': 2
    }

    if num_keypoints > 0:
        heads['kps'] = num_keypoints * 2

    return DLASeg(f'dla34', heads,
                  pretrained=False,
                  down_ratio=down_ratio,
                  final_kernel=1,
                  last_level=5,
                  head_conv=head_conv,
                  freeze_base=freeze_base,
                  rotated_boxes=rotated_boxes,
                  cfg=cfg)

def save_featuremap(tensor):
    np_array = tensor.cpu().detach().numpy()

    try:
        save_featuremap.counter += 1
    except AttributeError:
        save_featuremap.counter = 0

    try:
        os.mkdir('feature_map')
    except:
        pass

    batch_size = np_array.shape[0]

    for b in range(batch_size):
        img_list = []

        for f in np_array[b]:
            # np_img = np.interp(f, (f.min(), f.max()), (0, 255))
            # np_img = np_img.astype(np.uint8)
            # np_img = np.dstack((np_img, np.zeros_like(np_img), np.zeros_like(np_img)))
            # img_list.append(np_img)

            cmap = plt.cm.jet
            norm = plt.Normalize(vmin=f.min(), vmax=f.max())
            image = cmap(norm(f))
            img_list.append(image)
    
        # imageio.mimsave('{}/{:06d}.png'.format('feature_map', save_featuremap.counter), img_list, duration=1)
        cv2.imwrite('{}/{:06d}.png'.format('feature_map', save_featuremap.counter), (img_list[-1][:,:,[2,1,0,3]]*255).astype(np.uint8))
        save_featuremap.counter += 1
