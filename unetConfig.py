'''
    PATHS AND VARIABLES TO MAKE THE CODE MORE LEGIBLE AND ADAPTABLE
'''

import ETL.dataConfig as dataConfig

DATA_ROOT_DIR = dataConfig.DATA_ROOT_DIR

IMG_DIR = DATA_ROOT_DIR + "images/"
IMG_TRAIN_DIR = DATA_ROOT_DIR + "images/train/"
IMG_VAL_DIR = DATA_ROOT_DIR + "images/val/"

BINMASK_DIR = DATA_ROOT_DIR + "binMasks/"
BINMASK_TRAIN_DIR = DATA_ROOT_DIR + "binMasks/train/"
BINMASK_VAL_DIR = DATA_ROOT_DIR + "binMasks/val/"