# -*- coding:utf-8 -*-

from __future__ import absolute_import, division, print_function
import numpy as np
import math
from libs.configs import cfgs


def get_all_binay_label(num_label, angle_range):
    all_binay_label = []
    tmp = 10000000 if angle_range == 90 else 100000000
    for i in range(num_label):
        binay = bin(i+1)
        binay = int(binay.split('0b')[-1]) + tmp
        binay = np.array(list(str(binay)[1:]), np.int32)
        all_binay_label.append(binay)
    return np.array(all_binay_label)


def angle_binay_label(angle_label, angle_range):
    """
    :param angle_label: [-90,0) or [-90, 0)
    :param angle_range: 90 or 180
    :return:
    """
    angle_label = np.array(-np.round(angle_label), np.int32)
    inx = angle_label == 0
    angle_label[inx] = angle_range
    all_binay_label = get_all_binay_label(angle_range, angle_range)
    binay_label = all_binay_label[angle_label - 1]
    return np.array(binay_label, np.float32)


if __name__ == '__main__':
    binay_label = angle_binay_label([0, -1, -2, -45, -80, -90], 90)
    print(binay_label)