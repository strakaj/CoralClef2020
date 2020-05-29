#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Ivan Gruber"
__version__ = "1.0.0"
__maintainer__ = "Ivan Gruber"
__email__ = "ivan.gruber@seznam.cz"

"""
Saving annotations to json file in COCO format
"""

import cv2
import json

coco_annotation = {'images': [], 'categories': [],
                   'annotations': []}
types_of_substrates = ['c_hard_coral_branching', 'c_hard_coral_submassive', 'c_hard_coral_boulder',
                       'c_hard_coral_encrusting', 'c_hard_coral_table', 'c_hard_coral_foliose', 'c_hard_coral_mushroom',
                       'c_soft_coral', 'c_soft_coral_gorgonian', 'c_sponge', 'c_sponge_barrel', 'c_fire_coral_millepora',
                       'c_algae_macro_or_leaves']
dataset_path = "training_set_2020_flip/"
boxes_file = 'annotations_train_2020_boxes.txt'
polygons_file = 'annotations_train_2020_polygons.txt'

txtfile_b = open(boxes_file, 'r')
lines_b = txtfile_b.readlines()
txtfile_p = open(polygons_file,'r')
lines_p = txtfile_p.readlines()

category_id = 0
observation_id, observation_count, coral_class, _, x1, y2, x2, y1 = lines_b[0].split(" ")
pivot = observation_id
f = dataset_path + observation_id + '.jpg'
image = cv2.imread(f, 1)
ann_id = 0
img_id = 0
coco_annotation['images'].append({
    'file_name': observation_id + '.jpg',
    'url': "",
    'height': image.shape[0],
    'width': image.shape[1],
    'id': img_id
})

for line_b,line_p in zip(lines_b, lines_p):
    observation_id, observation_count, coral_class, _, x1, y2, x2, y1 = line_b.split(" ")
    polygon = line_p.split(" ")
    pol = polygon[4:]
    if pivot != observation_id:
        pivot = observation_id
        f = dataset_path + observation_id + '.jpg'
        image = cv2.imread(f, 1)
        img_id +=1
        coco_annotation['images'].append({
            'file_name': observation_id + '.jpg',
            'url': "",
            'height': image.shape[0],
            'width': image.shape[1],
            'id': img_id
        })
    bbox = [x1, y1.strip(), x2, y2]
    # image = image/255.
    for i in range(0, len(types_of_substrates)):
        if coral_class == types_of_substrates[i]:
            category_id = i
            break
    coco_annotation['annotations'].append({
        'segmentation': pol,
        'image_id': img_id,
        'id': ann_id,
        "category_id": category_id,
        "iscrowd": 0,
        'bbox': bbox,
        'area': ""

    })
    print('Annotation ' + str(ann_id)+ ' for img ' + str(img_id)+ ' is processed!')

    ann_id +=1

with open('annotation.json', 'w') as file:
    json.dump(coco_annotation, file)
