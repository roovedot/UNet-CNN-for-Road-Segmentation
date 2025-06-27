import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import unetConfig

class RoadSegDataset(Dataset):
    def __init__(self, image_dir=unetConfig.IMG_TRAIN_DIR, mask_dir=unetConfig.BINMASK_TRAIN_DIR, transform=None):
        #super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.images = os.listdir(image_dir) # Get list of image fileNames
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Get paths to the image and corresponding mask
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx]).replace('leftImg8bit', 'binMask')

        # Load image and mask
        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)
        mask[mask == 255.0] = 1.0 #Convert to (0-1) binMask because final activation is sigmoid

        # Apply transformations if provided
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        
        return image, mask
    


if __name__ == '__main__':
    dataset = RoadSegDataset()

    # aachen_000000_000019_leftImg8bit.png
    # aachen_000000_000019_binMask.png