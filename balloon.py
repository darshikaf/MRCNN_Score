import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import mrcnn.utils
import mrcnn.visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

from config import BalloonConfig, InferenceConfig
from dataset import BalloonDataset

config = BalloonConfig()
config = InferenceConfig()
DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0
TEST_MODE = "inference"
ROOT_DIR = "/content/gdrive/MyDrive/mcrnn"
DATASET_DIR = os.path.join(ROOT_DIR, "Mask_RCNN/balloon")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
MODEL = os.path.join(MODEL_DIR, "balloon20210408T0056/mask_rcnn_balloon_0030.h5")




def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


dataset = BalloonDataset()
dataset.load_balloon(DATASET_DIR, "val")
dataset.prepare()

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

model.load_weights(MODEL, by_name=True)

image_id = random.choice(dataset.image_ids)
image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
info = dataset.image_info[image_id]
print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                       dataset.image_reference(image_id)))

# Run object detection
results = model.detect([image], verbose=1)

# Display results
ax = get_ax(1)
r = results[0]
mrcnn.visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            dataset.class_names, r['scores'], ax=ax,
                            title="Predictions")
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)
