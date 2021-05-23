import numpy as np
import torch


def xywhc2label(bboxs, S, num_anchors, num_classes):
    # bboxs is a xywhc list: [(x,y,w,h,c),(x,y,w,h,c)....]
    label = np.zeros((S, S, 5 + num_classes))
    for x, y, w, h, c in bboxs:
        x_grid = int(x // (1.0 / S))
        y_grid = int(y // (1.0 / S))
        # xx = x / (1.0 / S) - x_grid
        # yy = y / (1.0 / S) - y_grid
        xx, yy = x, y
        label[y_grid, x_grid, 0:5] = np.array([xx, yy, w, h, 1])
        label[y_grid, x_grid, 5 + c] = 1
    return label


def calculate_iou(bbox1, bbox2, box_form):
    if box_form == 'xyxy':
        # bbox form: x1 y1 x2 y2
        bbox1, bbox2 = bbox1.cpu().detach().numpy().tolist(), bbox2.cpu().detach().numpy().tolist()

        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])  # bbox1's area
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])  # bbox2's area

        max_left = max(bbox1[0], bbox2[0])
        min_right = min(bbox1[2], bbox2[2])
        max_top = max(bbox1[1], bbox2[1])
        min_bottom = min(bbox1[3], bbox2[3])

        if max_left >= min_right or max_top >= min_bottom:
            return 0
        else:
            # iou = intersect / union
            intersect = (min_right - max_left) * (min_bottom - max_top)
            return intersect / (area1 + area2 - intersect)

    if box_form == 'xywh':
        # bbox form: x y w h
        bbox1, bbox2 = bbox1.cpu().detach().numpy().tolist(), bbox2.cpu().detach().numpy().tolist()

        area1 = bbox1[2] * bbox1[3]  # bbox1's area
        area2 = bbox2[2] * bbox2[3]  # bbox2's area

        max_left = max(bbox1[0] - bbox1[2] / 2, bbox2[0] - bbox2[2] / 2)
        min_right = min(bbox1[0] + bbox1[2] / 2, bbox2[0] + bbox2[2] / 2)
        max_top = max(bbox1[1] - bbox1[3] / 2, bbox2[1] - bbox2[3] / 2)
        min_bottom = min(bbox1[1] + bbox1[3] / 2, bbox2[1] + bbox2[3] / 2)

        if max_left >= min_right or max_top >= min_bottom:
            return 0
        else:
            # iou = intersect / union
            intersect = (min_right - max_left) * (min_bottom - max_top)
            return intersect / (area1 + area2 - intersect)


def calculate_iou2(bbox1, bbox2):
    # bbox: x y w h
    bbox1, bbox2 = bbox1.cpu().detach(), bbox2.cpu().detach()

    area1 = bbox1[:, 2] * bbox1[:, 3]  # bbox1's area
    area2 = bbox2[2] * bbox2[3]  # bbox2's area

    max_left = torch.max(bbox1[:, 0] - bbox1[:, 2] / 2, bbox2[0] - bbox2[2] / 2)
    min_right = torch.min(bbox1[:, 0] + bbox1[:, 2] / 2, bbox2[0] + bbox2[2] / 2)
    max_top = torch.max(bbox1[:, 1] - bbox1[:, 3] / 2, bbox2[1] - bbox2[3] / 2)
    min_bottom = torch.min(bbox1[:, 1] + bbox1[:, 3] / 2, bbox2[1] + bbox2[3] / 2)

    ret = ((min_right - max_left) * (min_bottom - max_top)) / (
            area1 + area2 - (min_right - max_left) * (min_bottom - max_top))

    # iou<0 -> 0
    keep = (ret < 0)
    idxs_select = torch.nonzero(keep).view(-1)
    ret[idxs_select] = 0

    return ret
