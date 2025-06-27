import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from UNet_model import UNet
import unetConfig
#from utils import load_checkpoint, save_checkpoint, get_loaders, check_accuracy, save_predictions_as_imgs)

# HYPERPARAMETERS
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 3
NUM_WORKERS = 8
IMAGE_HEIGHT = 1024 # Originally 1024
IMAGE_WIDTH = 2048 # Originally 2048
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_IMG_DIR = unetConfig.IMG_TRAIN_DIR
TRAIN_MASK_DIR = unetConfig.BINMASK_TRAIN_DIR
VAL_IMG_DIR = unetConfig.IMG_VAL_DIR
VAL_MASK_DIR = unetConfig.BINMASK_VAL_DIR

def train_fn(loader, model, optimizer, loss_fn, scaler):
    pass

def main():
    pass

if __name__ == '__main__':
    main() # Call main as a function to avoid problems in windows