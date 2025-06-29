import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from UNet_model import UNet
import unetConfig
from unetUtils import (load_checkpoint, save_checkpoint, get_loaders, check_accuracy, save_predictions_as_imgs)

# HYPERPARAMETERS
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 3
NUM_WORKERS = 0
IMAGE_HEIGHT = 256 # Originally 1024
IMAGE_WIDTH = 512 # Originally 2048
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_IMG_DIR = unetConfig.IMG_TRAIN_DIR
TRAIN_MASK_DIR = unetConfig.BINMASK_TRAIN_DIR
VAL_IMG_DIR = unetConfig.IMG_VAL_DIR
VAL_MASK_DIR = unetConfig.BINMASK_VAL_DIR

# Debug
print(torch.cuda.memory_summary())
print("Available devices:")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Device index: {torch.cuda.current_device()}")
    print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Memory Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
else:
    print("Using CPU")

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE) # Add channel dimension and convert to float

        # Forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():

    # Transformtaions forn train data
    train_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=15, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.015),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])

    # Transformations for val data
    val_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH), 
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])

    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss() # Binary Cross Entropy with Logits Loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS
    )

    scaler = torch.cuda.amp.GradScaler()  # For mixed precision training

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    check_accuracy(val_loader, model, device=DEVICE)  
    
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # Save model
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        save_checkpoint(checkpoint)

        # Check accuracy on validation set
        check_accuracy(val_loader, model, device=DEVICE)

        # Save predictions as images
        save_predictions_as_imgs(val_loader, model, device=DEVICE)
        


if __name__ == '__main__':
    main() # Call main as a function to avoid problems in windows
