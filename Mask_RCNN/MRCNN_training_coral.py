from os import listdir
import os
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from numpy import mean
from mrcnn.utils import Dataset
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
from mrcnn.model import MaskRCNN
from mrcnn.config import Config
from matplotlib import pyplot
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
from matplotlib.patches import Rectangle
import matplotlib.image as img
import matplotlib.pyplot as plt
import cv2 as cv
import skimage
import json
import numpy as np
from math import ceil



DATASET_DIR = 'Datasets/CoralClef2020/Data/training_set_2020/'
ANNOTATION_TRAIN = 'Datasets/CoralClef2020/Annotations/train_annot.json'
ANNOTATION_VAL = 'Datasets/CoralClef2020/Annotations/validation_annot.json'

CLASSES = ['c_hard_coral_branching', 'c_hard_coral_submassive', 'c_hard_coral_boulder',
           'c_hard_coral_encrusting', 'c_hard_coral_table', 'c_hard_coral_foliose', 'c_hard_coral_mushroom',
           'c_soft_coral', 'c_soft_coral_gorgonian', 'c_sponge', 'c_sponge_barrel', 'c_fire_coral_millepora',
           'c_algae_macro_or_leaves']

# cesta k modelu
MODEL_PATH = '/trained_weights/mask_rcnn_coco.h5'

# cesta pro ulozeni natrenovaneho modelu
MODEL_DIR =  '/training_summaries/'

EPOCHS = 200



# trida pro nacteni datasetu
class CoralDataset(Dataset):
    
    # nacteni obrazku
    def load_dataset(self, is_train=True):

        # pridani trid do datasetu
        for i in range(len(CLASSES)):
            self.add_class("dataset", i+1, CLASSES[i])
        
        # nastaveni cesty k souboru s anotacemi
        if(is_train):
            annotation_path = ANNOTATION_TRAIN
        else:
            annotation_path = ANNOTATION_VAL
            
        with open(annotation_path) as json_file:
            data = json.load(json_file)
        
        # nacteni anotaci a vlozeni obrazku do datasetu
        for annot in data['images']:
            image_id = annot['id']
            img_path = DATASET_DIR + annot['file_name']
            ann_path = annotation_path

            self.add_image('dataset', 
                           image_id=image_id, 
                           width=annot['width'], height=annot['height'],
                           path=img_path, 
                           annotation=ann_path)
            


    # nacteni masek
    def load_mask(self, image_id):
        # nacteni informaci o obrazku
        info = self.image_info[image_id]

        # nacteni anotaci
        path = info['annotation']
        with open(path) as json_file:
            data = json.load(json_file)
        
        segments = []
        class_ids = []
        for s in data['annotations']:
            if(s['image_id'] == info['id']):
                segments.append(s['segmentation'])
                class_ids.append(s['category_id'] + 1)
            
        # vytvoreni 2d pole pro kazdou masku
        masks = zeros([info['height'], info['width'], len(segments)], dtype='uint8')
        for i in range(len(segments)):
            r = []
            c = []
            for s0,s1 in zip(segments[i][0::2], segments[i][1::2]):
                r.append(int(s1))
                c.append(int(s0))
            r = np.array(r)
            c = np.array(c)
            rr, cc = skimage.draw.polygon(r, c)
            masks[rr, cc, i] = 1
        return masks, asarray(class_ids, dtype='int32')
 
    # nacteni informaci o obrazku
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']        


 
# nacteni trenovaci mnoziny
train_set = CoralDataset()
train_set.load_dataset(is_train=True)
train_set.prepare()
print('Trenovaci: %d' % len(train_set.image_ids))

# nacteni validacni mnoziny
val_set = CoralDataset()
val_set.load_dataset(is_train=False)
val_set.prepare()
print('Validacni: %d' % len(val_set.image_ids))




# nastaveni modelu
class CoralConfig(Config):
    NAME = "CoralClef2020"
    NUM_CLASSES = 13 + 1
    BATCH_SIZE = 2
    # Number of training steps per epoch
    STEPS_PER_EPOCH = ceil(len(train_set.image_ids)/BATCH_SIZE)
    LEARNING_RATE = 0.00001


config = CoralConfig()
config.display()



# definovani modelu
model = MaskRCNN(mode='training', model_dir=MODEL_DIR, config=config)

# pokud je pouzit predtrenovany model s jinym poctem trid, posledni vrstvy jsou inicializovany nahodne
model.load_weights(MODEL_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
# pokud pokracujeme v trenovani modelu
# model.load_weights(MODEL_PATH, by_name=True)

            

model.train(train_set, val_set, 
            learning_rate=config.LEARNING_RATE,
            epochs=EPOCHS, 
            layers="all")
