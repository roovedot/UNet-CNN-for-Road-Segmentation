import torch
import torchvision
from dataset import RoadSegDataset
from torch.utils.data import DataLoader
import unetConfig

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    

def get_loaders(
        train_img_dir,
        train_mask_dir,
        val_img_dir,
        val_mask_dir,
        batch_size,
        train_transform,
        val_transform,
        num_workers,
        pin_memory=True
):
    """ Returns DataLoaders for training and validation datasets."""

    # Creates the training dataset
    train_ds = RoadSegDataset(
        image_dir=train_img_dir,
        mask_dir=train_mask_dir,
        transform=train_transform
    )

    # Creates the validation dataset
    val_ds = RoadSegDataset(
        image_dir=val_img_dir,
        mask_dir=val_mask_dir,
        transform=val_transform
    )
    
    # Creates the training dataloader
    train_loader = DataLoader(
        train_ds,
        batch_size = batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    # Creates the validation dataloader
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    """ Checks the accuracy of the model on the given dataset loader."""
    
    num_correct = 0
    num_pixels = 0
    dice_score = 0.0

    model.eval()  # Set model to evaluation mode

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device).unsqueeze(1)  

            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()  # Convert to binary
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)

            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8) # Just for binary
    
    print(f"Got {num_correct} / {num_pixels} with accuracy: {num_correct / num_pixels:.4f}")

    print(f"Dice Score: {dice_score / len(loader):.4f}")

    model.train()  # Set model back to training mode 


def save_predictions_as_imgs(loader, model, folder=unetConfig.DATA_ROOT_DIR+"comparePreds/", device="cuda"): 
    """ Saves the model predictions as images."""
    
    model.eval()  # Set model to evaluation mode

    # 
    for idx, (x, y) in enumerate(loader): # Iterate through the data loader
        x = x.to(device=device) # Move input to the specified device
        with torch.no_grad(): # Disable gradient calculation
            preds = torch.sigmoid(model(x)) # Get model predictions
            preds = (preds > 0.5).float()  # Convert to binary

        torchvision.utils.save_image(
            preds, f"{folder}pred_{idx}.png"
        )
        torchvision.utils.save_image(
            y.unsqueeze(1), f"{folder}y_{idx}.png"
        )


    model.train()  # Set model back to training mode