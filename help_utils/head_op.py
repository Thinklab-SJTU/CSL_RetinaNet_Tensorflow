# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import tensorflow as tf


def get_head_quadrant(head, gtbox):
    """
    :param head: [head_x, head_y]
    :param gtbox: [x_c, y_c, w, h, theta, label]
    :return: head quadrant 0/1/2/3
    """
    head_quadrant = []
    for i, box in enumerate(gtbox):
        detla_x = head[i][0] - box[0]
        detla_y = head[i][1] - box[1]
        if (detla_x >= 0) and (detla_y >= 0):
            head_quadrant.append(0)
        elif (detla_x >= 0) and (detla_y <= 0):
            head_quadrant.append(1)
        elif (detla_x <= 0) and (detla_y <= 0):
            head_quadrant.append(2)
        else:
            head_quadrant.append(3)
    return np.array(head_quadrant, np.int32)


def get_head(gtboxes_and_label_batch):
    """
    :param gtboxes_and_label_batch: [x1, y1, x2, y2, x3, y3, x4, y4, head_x, head_y, label]
    :return: [x1, y1, x2, y2, x3, y3, x4, y4, label], [head_x, head_y]
    """
    x1, y1, x2, y2, x3, y3, x4, y4, head_x, head_y, label = tf.unstack(gtboxes_and_label_batch, axis=1)
    coords_label = tf.transpose(tf.stack([x1, y1, x2, y2, x3, y3, x4, y4, label]))
    head = tf.transpose(tf.stack([head_x, head_y]))

    return coords_label, head