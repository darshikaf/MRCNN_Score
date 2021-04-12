import os
from typing import List

import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

import mrcnn.model as modellib

from mrcnn.utils import resize_image
from mrcnn.visualize import display_images, display_instances

import constants

from config import BalloonConfig, InferenceConfig
from dataset import BalloonDataset
from utils import add_class, get_ax, load_image


# TODO: add an image_info dict for image statistics

class Predict(object):
    def __init__(self, image_id: str, classes: List):
        self.image_id = image_id
        self.classes = classes
        self.config = InferenceConfig()
        self.device = "/cpu:0"

    def prep_data(self):
        # TODO: use utils.Dataset instead
        dataset = BalloonDataset()
        for index, _cls in enumerate(self.classes):
            dataset.add_class("balloon", index+1, str(_cls))
            # TODO: Add logging
        dataset.prepare()

    def load_model(self):
        with tf.device(self.device):
            model = modellib.MaskRCNN(
                mode="inference",
                model_dir=constants.MODEL_DIR,
                config=self.config
            )
        model.load_weights(constants.MODEL, by_name=True)
        return model

    def predict(self):
        try:
            self.prep_data()
            model = self.load_model()
        except Exception as e:
            raise e
        image_path = os.path.join(constants.INPUT_DATA, image_id)
        image = load_image(image_path)
        height, width = image.shape[:2]
        image, window, scale, padding, crop = resize_image(
            image,
            min_dim=self.config.IMAGE_MIN_DIM,
            min_scale=self.config.IMAGE_MIN_SCALE,
            max_dim=self.config.IMAGE_MAX_DIM,
            mode=self.config.IMAGE_RESIZE_MODE
        )
        results = model.detect([image], verbose=1)
        ax = get_ax(1)
        r = results[0]
        display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            self.classes, r['scores'], ax=ax,
                            title="Predictions")
        detected_image_name = f"detect_{self.image_id.split('.')[0]}.png"
        stored_path = f"{constants.OUTPUT_DATA}/{detected_image_name}"
        # TODO: logs
        plt.savefig(stored_path)
        return stored_path


filename = "5603212091_2dfe16ea72_b.jpg"
prediction = Predict(filename, ["balloon"])
stored_path = prediction.predict()