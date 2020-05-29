#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Ivan Gruber"
__version__ = "1.0.0"
__maintainer__ = "Ivan Gruber"
__email__ = "ivan.gruber@seznam.cz"

"""
Flip wrongly rotated images
"""

import cv2

dataset_path = "training_set_2020/"
output_path = "training_set_2020_flip/"
boxes_file = 'annotations_train_2020_boxes.txt'

poc = 1
txtfile = open(boxes_file, 'r')
lines = txtfile.readlines()
txtfile.close()
observation_id, observation_count, coral_class, _, x1, y2, x2, y1 = lines[0].split(" ")
pivot = observation_id
f = dataset_path + observation_id + '.jpg'
image = cv2.imread(f, 1)
cv2.imwrite(output_path + pivot + ".jpg", image)

for line in lines:
    observation_id, observation_count, coral_class, _, x1, y2, x2, y1 = line.split(" ")
    if pivot != observation_id:
        poc +=1
        pivot = observation_id
        f = dataset_path + observation_id + '.jpg'
        image = cv2.imread(f, 1)
        if poc >= 319:
            image = cv2.flip(image, 0)
            image = cv2.flip(image, 1)
            print(f+" was flipped!")
        cv2.imwrite(output_path + pivot + ".jpg", image)
        print(f + ' ready!')

cv2.imwrite(output_path + pivot + ".jpg", image)
print(f + ' ready!')