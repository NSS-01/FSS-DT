import numpy as np
import math

def read_points_from_file(file_name, split_char= '\t'):
    points = []
    with open(file_name) as fr:
        for line in fr.readlines():
            cur_line = line.strip().split(split_char)
            cur_line = cur_line[:2]
            flt_line = list(map(float, cur_line))
            points.append(flt_line)
    return points


def read_points_and_label_from_file(file_name, split_char= '\t'):
    points = []
    labels = []
    with open(file_name) as fr:
        for line in fr.readlines():
            cur_line = line.strip().split(split_char)
            cur_line = cur_line[:2]
            label = cur_line[1]
            flt_line = list(map(float, cur_line))
            label = int(label)
            points.append(flt_line)
            labels.append(label)
    return points, labels

