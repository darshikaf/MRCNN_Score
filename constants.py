import os

ROOT_DIR = "/content/gdrive/MyDrive/mcrnn"
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
MODEL = os.path.join(MODEL_DIR, "balloon20210409T0634/mask_rcnn_balloon_0030.h5")
INPUT_DATA = os.path.join(ROOT_DIR, "MRCNN_Score/")
OUTPUT_DATA = os.path.join(ROOT_DIR, "MRCNN_Score/outputs")