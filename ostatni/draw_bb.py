#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Ivan Gruber"
__version__ = "1.0.0"
__maintainer__ = "Ivan Gruber"
__email__ = "ivan.gruber@seznam.cz"

"""
Draw bounding boxes
"""

import cv2

dataset_path = "training_set_2020_flip/"
output_path = "rect/"
boxes_file = 'annotations_train_2020_boxes.txt'

thickness = 2
color = (0, 0, 255)
poc = 1
txtfile = open(boxes_file, 'r')
lines = txtfile.readlines()
txtfile.close()
observation_id, observation_count, coral_class, _, x1, y2, x2, y1 = lines[0].split(" ")
pivot = observation_id
f = dataset_path + observation_id + '.jpg'
image = cv2.imread(f, 1)

for line in lines:
    observation_id, observation_count, coral_class, _, x1, y2, x2, y1 = line.split(" ")
    if pivot != observation_id:
        poc +=1
        cv2.imwrite(output_path + pivot + ".jpg", image)
        print(f + ' ready!')
        pivot = observation_id
        f = dataset_path + observation_id + '.jpg'
        image = cv2.imread(f, 1)
        # #originalni data jsou od snimku c.319 otocena o 180 stupnu (vzhledem k anotacim), je proto je treba spravne natocit
        # if poc >= 319:
        #     image = cv2.flip(image, 0)
        #     image = cv2.flip(image, 1)
        # #------------------------------------------------------------------------------------------------------------------
    start_point = (int(x1), int(y1))
    end_point = (int(x2), int(y2.strip()))
    image = cv2.rectangle(image, start_point, end_point, color, thickness)
cv2.imwrite(output_path + pivot + ".jpg", image)
print(f + ' ready!')