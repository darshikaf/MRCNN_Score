import os

ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "model")
MODEL = os.path.join(MODEL_DIR, "model/mask_rcnn_balloon_0030.h5")
INPUT_DATA = os.path.join(ROOT_DIR, "inputs")
OUTPUT_DATA = os.path.join(ROOT_DIR, "outputs")