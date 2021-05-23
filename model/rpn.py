import math

import torch
import torch.nn.functional as F
from torch import nn

from utils import calculate_iou


class RPN(nn.Module):
    def __init__(self, training):
        super(RPN, self).__init__()
        self.conv3x3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv1x1_cls = nn.Conv2d(512, 18, 1)
        self.conv1x1_loc = nn.Conv2d(512, 36, 1)
        self.training = training

        # generate anchors
        self.anchors = generate_anchors(50, 37, base_size=8, dsr=16)  # 16650*4
        print('anchors:', self.anchors)

    def forward(self, feats, gt_boxes, im_info):
        batch_size = feats.size(0)

        # feature map after conv layer
        rpn_conv1 = F.relu(self.conv3x3(feats), inplace=True)

        # rpn classification score
        rpn_cls_score = self.conv1x1_cls(rpn_conv1)  # b*18*50*37
        rpn_cls_score_reshape = reshape(rpn_cls_score, 2)  # b*2*450*37
        rpn_cls_score_softmax = F.softmax(rpn_cls_score_reshape, 1)  # b*2*450*37
        rpn_cls_score_reshape_back = reshape(rpn_cls_score_softmax, 18)  # b*18*50*37

        # rpn offsets to the anchor boxes
        rpn_loc_pred = self.conv1x1_loc(rpn_conv1)  # b*36*50*37

        # ----------------------------------------generate proposals----------------------------------------------------
        rpn_proposals = roi_pooling(rpn_cls_score_softmax, rpn_loc_pred, im_info[1])

        # rpn loss
        rpn_loss_cls = 0
        rpn_loss_loc = 0

        if self.training:
            assert gt_boxes is not None

            anchors = self.anchors

            # keep only inside anchors
            keep = ((anchors[:, 0] >= 0) &
                    (anchors[:, 1] >= 0) &
                    (anchors[:, 2] <= int(im_info[0][1]) + 0) &
                    (anchors[:, 3] <= int(im_info[0][0]) + 0))
            idxs_inside = torch.nonzero(keep).view(-1)
            anchors = anchors[idxs_inside, :]  # 5076*4
            print('anchors after clip:', anchors.shape, anchors)

            # ----------------------------------compute classification loss---------------------------------------------

            # b*450*37*2 -> b*16650*2
            rpn_cls_score_reshape = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
            rpn_cls_score_reshape = rpn_cls_score_reshape[:, idxs_inside, :]  # b*5076*2

            cls_labels = torch.zeros((batch_size, anchors.shape[0])).long()
            positives = []
            positive_idxs = []
            for b in range(batch_size):
                positive_idx_gt = {}
                iou_matrix = torch.zeros((anchors.shape[0], len(gt_boxes[b])))  # 5076*num_gts_per_img

                # compute iou matrix
                for i in range(anchors.shape[0]):
                    for j in range(len(gt_boxes[b])):
                        iou = calculate_iou(anchors[i], gt_boxes[b][j], box_form='xyxy')
                        iou_matrix[i][j] = iou

                # 1.for each anchor if iou < 0.3, negative
                for i in range(anchors.shape[0]):
                    max_iou = torch.max(iou_matrix[i][:])
                    if max_iou < 0.3:
                        cls_labels[b][i] = 0

                # 2.for each gt, max iou, positive
                idxs = torch.max(iou_matrix[:][:], 0)[1]
                cls_labels[b][idxs] = 1
                for i, idx in enumerate(idxs):
                    if idx not in positive_idx_gt:
                        positive_idx_gt[idx] = i
                        positive_idxs.append(b * idx + idx)

                # 3.for each anchor if iou > 0.7, positive
                for i in range(anchors.shape[0]):
                    row = torch.zeros((1, len(gt_boxes[b])))
                    row[0] = iou_matrix[i][:]
                    max_iou, max_idx = torch.max(row, dim=1)
                    if max_iou > 0.7:
                        positive_idx_gt[i] = max_idx
                        positive_idxs.append(i * b + i)
                        cls_labels[b][i] = 1

                positives.append(positive_idx_gt)

            rpn_cls_score_reshape = rpn_cls_score_reshape.view(-1, 2)
            cls_labels = cls_labels.view(-1)
            print(rpn_cls_score_reshape.shape, cls_labels.shape)
            rpn_loss_cls += F.cross_entropy(rpn_cls_score_reshape, cls_labels)

            # -------------------------------------compute regression loss----------------------------------------------
            print(rpn_loc_pred.shape)
            # b*36*50*37 -> b*16650*4
            rpn_loc_pred = rpn_loc_pred.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)
            rpn_loc_pred = rpn_loc_pred[:, idxs_inside, :]  # b*5076*4
            loc_labels = torch.zeros((batch_size, anchors.shape[0], 4)).long()

            for b in range(batch_size):
                for i in range(anchors.shape[0]):
                    if i in positives[b]:
                        loc_labels[b][i][:] = gt_boxes[b][positives[i]]

            rpn_loc_pred = rpn_loc_pred[:, positive_idxs, :]
            loc_labels = loc_labels[:, positive_idxs, :]

            rpn_loc_pred = rpn_loc_pred.view(-1, 4)
            loc_labels = loc_labels.view(-1, 4)

            rpn_loss_loc = F.smooth_l1_loss(rpn_loc_pred, loc_labels, reduction='sum') / 256

        print(rpn_loss_cls, rpn_loss_loc)
        return rpn_proposals, rpn_loss_cls, rpn_loss_loc


def generate_anchors2(feat_x, feat_y, base_size, dsr):
    # start = time.time()
    anchors = torch.zeros((feat_x, feat_y, 5, 4))  # 50*37*9*4
    for x in range(feat_x):
        for y in range(feat_y):
            for i, size in enumerate([base_size, base_size * 2, base_size * 4]):
                anchors[x, y, i * 3] = torch.tensor(
                    [(x - size / 2) * dsr, (y - size / 2) * dsr, (x + size / 2) * dsr, (y + size / 2) * dsr])

                short = math.sqrt((size ** 2) / 2)

                anchors[x, y, i * 3 + 1] = torch.tensor(
                    [(x - short) * dsr, (y - short / 2) * dsr, (x + short) * dsr, (y + short / 2) * dsr])

                anchors[x, y, i * 3 + 2] = torch.tensor(
                    [(x - short / 2) * dsr, (y - short) * dsr, (x + short / 2) * dsr, (y + short) * dsr])

    anchors = torch.reshape(anchors, (-1, 4))
    anchors[:, 0] = (anchors[:, 0] + anchors[:, 2]) / 2 / 800
    anchors[:, 1] = (anchors[:, 1] + anchors[:, 3]) / 2 / 600
    anchors[:, 2] = (anchors[:, 2] - anchors[:, 0]) / 800
    anchors[:, 3] = (anchors[:, 3] - anchors[:, 1]) / 600
    # print(time.time() - start)
    return anchors


def generate_anchors(feat_x, feat_y, base_size, dsr):
    anchors = []  # 50*37*9*4

    for x in range(feat_x):
        anchors_x = []  # 37*9*4
        for y in range(feat_y):
            anchors_xy = []  # 9*4

            # three size: 16 32 64
            for size in [base_size, 2 * base_size, 4 * base_size]:
                # three ratios: 1:1 1:2 2:1
                # anchors_xy.append([(x - size / 2) * dsr, (y - size / 2) * dsr,
                #                    (x + size / 2) * dsr, (y + size / 2) * dsr])
                # anchors_xy.append([(x - size) * dsr, (y - size / 2 / 2) * dsr,
                #                    (x + size) * dsr, (y + size / 2 / 2) * dsr])
                # anchors_xy.append([(x - size / 2 / 2) * dsr, (y - size) * dsr,
                #                    (x + size / 2 / 2) * dsr, (y + size) * dsr])

                anchors_xy.append(
                    [(x - size / 2) * dsr, (y - size / 2) * dsr, (x + size / 2) * dsr, (y + size / 2) * dsr])

                short = math.sqrt((size ** 2) / 2)

                anchors_xy.append([(x - short) * dsr, (y - short / 2) * dsr,
                                   (x + short) * dsr, (y + short / 2) * dsr])
                anchors_xy.append([(x - short / 2) * dsr, (y - short) * dsr,
                                   (x + short / 2) * dsr, (y + short) * dsr])

            # if x == feat_x//2 and y == feat_y//2:
            #     print(anchors_xy)

            anchors_x.append(anchors_xy)
        anchors.append(anchors_x)

    # convert to tensor
    anchors = torch.Tensor(anchors)
    anchors = torch.reshape(anchors, (-1, 4))
    return anchors


def reshape(x, d):
    input_shape = x.size()
    x = x.view(
        input_shape[0],
        int(d),
        int(float(input_shape[1] * input_shape[2]) / float(d)),
        input_shape[3]
    )
    return x


def roi_pooling(rpn_cls_score_softmax, rpn_loc_pred, img):
    batch_size = rpn_cls_score_softmax.shape[0]
    for b in range(batch_size):
        pass
    return 0


def main():
    rpn = RPN(training=True)
    rpn.forward(torch.randn((10, 512, 50, 37)), torch.tensor([[[0, 0, 100, 100], [100, 100, 200, 200]]] * 10),
                [[600, 800], []])  # gt_boxes: b*n*4


if __name__ == "__main__":
    main()
