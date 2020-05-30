import h5py
import numpy as np
import shutil

from misc_utils.tensor_sampling_utils import sample_tensors
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt

from models.keras_ssd512 import ssd_512
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms


# nastaveni modelu
img_height = 512 # Height of the input images
img_width = 512 # Width of the input images
img_channels = 3 # Number of color channels of the input images
subtract_mean = [123, 117, 104] # The per-channel mean of the images in the dataset
swap_channels = [2, 1, 0] # The color channel order in the original SSD is BGR, so we should set this to `True`, but weirdly the results are better without swapping.
# TODO: Set the number of classes.
n_classes = 13 # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
scales = [0.07, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05] # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets.
# scales = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05] # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets.
aspect_ratios = [[1.0, 2.0, 0.5],
                [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                [1.0, 2.0, 0.5],
                [1.0, 2.0, 0.5]] # The anchor box aspect ratios used in the original SSD300; the order matters
two_boxes_for_ar1 = True
steps = [8, 16, 32, 64, 128, 256, 512] # The space between two adjacent anchor box center points for each predictor layer.
offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5] # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
clip_boxes = False # Whether or not you want to limit the anchor boxes to lie entirely within the image boundaries
variances = [0.1, 0.1, 0.2, 0.2] # The variances by which the encoded target coordinates are scaled as in the original implementation
normalize_coords = True


# pri trenovani z predronavanych vah

K.clear_session()

model = ssd_512(image_size=(img_height, img_width, img_channels),
                n_classes=n_classes,
                mode='training',
                l2_regularization=0.0005,
                scales=scales,
                aspect_ratios_per_layer=aspect_ratios,
                two_boxes_for_ar1=two_boxes_for_ar1,
                steps=steps,
                offsets=offsets,
                clip_boxes=clip_boxes,
                variances=variances,
                normalize_coords=normalize_coords,
                subtract_mean=subtract_mean,
                swap_channels=swap_channels)

print("Model built.")




weights_path = 'trained_weights/VGG_coco_SSD_512x512_iter_360000_coral.h5'


model.load_weights(weights_path, by_name=True)


print("Weights file loaded:", weights_path)


adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
model.compile(optimizer=adam, loss=ssd_loss.compute_loss, metrics=['accuracy'])




# pri restartovani trenovani
'''
model_path =  'training_summaries/ssd512/ssd512_coralclef2020_epoch-200.h5'

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
K.clear_session() #

model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                               'L2Normalization': L2Normalization,
                                               'compute_loss': ssd_loss.compute_loss})

'''



# nacteni obrazku a anotaci
train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path = None)
val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path = None)



CoralClef_2020_images_dir      = '../CoralClef2020/training_set_2020/'


CoralClef_2020_annotations_train      =  '../CoralClef2020/annotations/train_annot.json'
CoralClef_2020_annotations_val        =  '../CoralClef2020/annotations/validation_annot.json'



classes = ['c_hard_coral_branching', 'c_hard_coral_submassive', 'c_hard_coral_boulder',
           'c_hard_coral_encrusting', 'c_hard_coral_table', 'c_hard_coral_foliose', 'c_hard_coral_mushroom',
           'c_soft_coral', 'c_soft_coral_gorgonian', 'c_sponge', 'c_sponge_barrel', 'c_fire_coral_millepora',
           'c_algae_macro_or_leaves']


train_dataset.parse_json(images_dirs=[CoralClef_2020_images_dir],
                         annotations_filenames=[CoralClef_2020_annotations_train],
                         ground_truth_available=True,
                         include_classes='all',
                         verbose = True,
                         ret=True
                        )

val_dataset.parse_json(images_dirs=[CoralClef_2020_images_dir],
                       annotations_filenames=[CoralClef_2020_annotations_val],
                       ground_truth_available=True,
                       include_classes='all',
                       verbose = True,
                       ret=True
                      )





batch_size = 2
ssd_data_augmentation = SSDDataAugmentation(img_height=img_height,
                                            img_width=img_width)
convert_to_3_channels = ConvertTo3Channels()
resize = Resize(height=img_height, width=img_width)




predictor_sizes = [model.get_layer('conv4_3_norm_mbox_conf').output_shape[1:3],
                   model.get_layer('fc7_mbox_conf').output_shape[1:3],
                   model.get_layer('conv6_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv7_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv8_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv9_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv10_2_mbox_conf').output_shape[1:3]]

ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                    img_width=img_width,
                                    n_classes=n_classes,
                                    predictor_sizes=predictor_sizes,
                                    scales=scales,
                                    aspect_ratios_per_layer=aspect_ratios,
                                    two_boxes_for_ar1=two_boxes_for_ar1,
                                    steps=steps,
                                    offsets=offsets,
                                    clip_boxes=clip_boxes,
                                    variances=variances,
                                    matching_type='multi',
                                    pos_iou_threshold=0.5,
                                    neg_iou_limit=0.5,
                                    normalize_coords=normalize_coords)



train_generator = train_dataset.generate(batch_size=batch_size,
                                         shuffle=True,
                                         transformations=[ssd_data_augmentation],
                                         label_encoder=ssd_input_encoder,
                                         returns={'processed_images',
                                                  'encoded_labels'},
                                         keep_images_without_gt=False)

val_generator = val_dataset.generate(batch_size=batch_size,
                                     shuffle=False,
                                     transformations=[resize],
                                     label_encoder=ssd_input_encoder,
                                     returns={'processed_images',
                                              'encoded_labels'},
                                     keep_images_without_gt=False)


train_dataset_size = train_dataset.get_dataset_size()
val_dataset_size   = val_dataset.get_dataset_size()

print("Trenovaci mnozina:\t{:>6}".format(train_dataset_size))
print("Validacni mnozina:\t{:>6}".format(val_dataset_size))




# nastaveni konstanty uceni
def lr_schedule(epoch):
    if epoch < 100:
        return 0.0001
    else:
        return 0.00001






model_checkpoint = ModelCheckpoint(filepath= 'training_summaries/ssd512/ssd512_coralclef2020_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=False,
                                   save_weights_only=False,
                                   mode='auto',
                                   period=1)


csv_logger = CSVLogger(filename= 'training_summaries/ssd512/ssd512_coralclef2020_training_log_1.csv',
                       separator=',',
                       append=True)

learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule,
                                                verbose=1)

terminate_on_nan = TerminateOnNaN()

callbacks = [model_checkpoint,
             csv_logger,
             learning_rate_scheduler,
             terminate_on_nan]






initial_epoch   = 0
final_epoch     = 200
steps_per_epoch = ceil(train_dataset_size/batch_size)

model.fit_generator(generator=train_generator,
                              steps_per_epoch=steps_per_epoch,
                              epochs=final_epoch,
                              callbacks=callbacks,
                              validation_data=val_generator,
                              validation_steps=ceil(val_dataset_size/batch_size),
                              initial_epoch=initial_epoch)





