import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import resnet34, resnet18, resnet50


def generate_anchors(feat_x, feat_y):
    # feat_x,feat_y 50*37
    base_size = 8
    dsr = 16  # down sample rate
    anchors = []  # 50*37*9*4

    for x in range(feat_x):
        anchors_x = []  # 37*9*4
        for y in range(feat_y):
            anchors_xy = []  # 9*4

            # three size: 16 32 64
            for size in [base_size, 2*base_size, 4*base_size]:

                # three ratios: 1:1 1:2 2:1
                anchors_xy.append([(x-size/2)*dsr, (y-size/2)*dsr,
                                   (x+size/2)*dsr, (y+size/2)*dsr])  # 4
                anchors_xy.append([(x-size)*dsr, (y-size/2/2)*dsr,
                                   (x+size)*dsr, (y+size/2/2)*dsr])
                anchors_xy.append([(x-size/2/2)*dsr, (y-size)*dsr,
                                   (x+size/2/2)*dsr, (y+size)*dsr])

            # if x == feat_x//2 and y == feat_y//2:
            #     print(anchors_xy)

            anchors_x.append(anchors_xy)

        anchors.append(anchors_x)

    # convert to tensor
    anchors = torch.Tensor(anchors)
    anchors = torch.reshape(anchors, (-1, 4))
    return anchors


def calculate_iou(bbox1, bbox2):
    # bbox: x1 y1 x2 y2
    bbox1, bbox2 = bbox1.cpu().detach().numpy(
    ).tolist(), bbox2.cpu().detach().numpy().tolist()

    area1 = (bbox1[2]-bbox1[0])*(bbox1[3]-bbox1[1])  # bbox1's area
    area2 = (bbox2[2]-bbox2[0])*(bbox2[3]-bbox2[1])  # bbox2's area

    max_left = max(bbox1[0], bbox2[0])
    min_right = min(bbox1[2], bbox2[2])
    max_top = max(bbox1[1], bbox2[1])
    min_bottom = min(bbox1[3], bbox2[3])

    if max_left >= min_right or max_top >= min_bottom:
        return 0
    else:
        # iou = intersect / union
        intersect = (min_right-max_left)*(min_bottom-max_top)
        return (intersect / (area1+area2-intersect))


class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()
        self.conv3x3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv1x1_cls = nn.Conv2d(512, 18, 1)
        self.conv1x1_bbox = nn.Conv2d(512, 36, 1)
        self.training = True

    def forward(self, feats, gt_boxes, im_info):
        batch_size = feats.size(0)
        # generate anchors
        anchors = generate_anchors(50, 37)  # 16650*4
        # print(anchors)

        # feature map after conv layer
        rpn_conv1 = F.relu(self.conv3x3(feats), inplace=True)

        # rpn classification score
        rpn_cls_score = self.conv1x1_cls(rpn_conv1)  # b*18*50*37
        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)  # b*2*450*37
        rpn_cls_prob_reshape = F.softmax(
            rpn_cls_score_reshape, 1)  # b*2*450*37
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, 18)  # b*18*50*37

        # rpn offsets to the anchor boxes
        rpn_bbox_pred = self.conv1x1_bbox(rpn_conv1)  # b*36*50*37

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

        if self.training:
            assert gt_boxes is not None

            keep = ((anchors[:, 0] >= 0) &
                    (anchors[:, 1] >= 0) &
                    (anchors[:, 2] < int(im_info[0][1]) + 0) &
                    (anchors[:, 3] < int(im_info[0][0]) + 0))

            inds_inside = torch.nonzero(keep, as_tuple=False).view(-1)

            # keep only inside anchors
            # anchors = anchors[inds_inside, :]  # 5576*4
            # print(anchors.shape)

            # compute cls loss
            rpn_cls_score = rpn_cls_score_reshape.permute(
                0, 2, 3, 1).contiguous().view(batch_size, -1, 2)  # b*16650*2

            for b in range(batch_size):

                iou_matrix = torch.zeros((anchors.shape[0], len(gt_boxes[b]))) # 16650*2

                labels=torch.zeros((anchors.shape[0]))

                # compute iou matrix
                for i in range(anchors.size(0)):
                    if i not in inds_inside:
                        continue
                    for j in range(len(gt_boxes[b])):
                        iou = calculate_iou(
                            anchors[i], torch.tensor(gt_boxes[b][j]))
                        iou_matrix[i][j] = iou
                
                # 1.for each anchor if iou < 0.3, negative
                for i in range(anchors.size(0)):
                    if i not in inds_inside:
                        continue
                    if torch.max(iou_matrix[i,:]) <0.3:
                        labels[i]=0
                    
                # 2.for each gt max iou, positive
                idxs=torch.max(iou_matrix[:][j],0)[1]
                labels[idxs]=1

                # 3.for each anchor if iou >0.7, positive
                for i in range(anchors.size(0)):
                    if i not in inds_inside:
                        continue
                    if torch.max(iou_matrix[i,:]) >0.7:
                        labels[i]=0

                


            # compute reg loss

        return feats

    def reshape(self, x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x


def main():
    rpn = RPN()
    rpn.forward(torch.ones((10, 512, 50, 37)), [
                [[1, 1, 100, 100], [50, 50, 200, 200]]]*10, [[600, 800]])  # gt_boxes: b*x*4


if __name__ == "__main__":
    main()
