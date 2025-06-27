from PIL import Image
import numpy as np
import dataConfig as dataConfig
import os
import re

def mask_image(img_path, out_dir, out_name, targetValue = 7, debug = False):
    """
    Takes a multiclass label_id mask image and creates a binary mask for a specific target value.
    
    Made for Cityscapes dataset. See class mapping: https://www.notion.so/V2-Custom-Image-Segmentation-for-Autonomous-Driving-Applications-149d04bb941780bead02c5130035e065?source=copy_link#151d04bb941780349a28e46ec19f80ce

    :param img_path: Path to the input image (multiclass mask).
    :param out_dir: Directory where the binary mask will be saved.
    :param out_name: Name of the output binary mask file (without path)(full name WITHOUT extension, binMask will be png).
    :param targetValue: The class value to create a binary mask for (default is 7, which is 'road').
    :param debug: If True, will print debug information and show the original and binary masks
    """
    
    # Load image (multiclass mask)
    try:
        originalMask = np.array(Image.open(img_path))
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return

    if debug:
        # should be similar to the comments
        print(f"Original mask shape: {originalMask.shape}") # (1024, 2048)
        print("OG mask dtype: ", originalMask.dtype) # uint8
        print(f"Original mask unique values: {np.unique(originalMask)}") # [ 0  1  3  4  7  8 11 17 20 21 22 23 24 25 26 33]
    
    # Create binary mask where targetValue is 255 and others are 0
        # check each pixel (element of the np.array) for ==targetValue and set true or false (1 or 0)
        # multiply by 255 so the mask is (0 - 255) instead of (0 - 1)
    try:
        binaryMask = (originalMask == targetValue).astype(np.uint8) * 255
    except Exception as e:
        print(f"Could not create binary mask for {img_path}: {e}")
        return

    if debug:
        # should be similar to the comments
        print(f"Original mask shape: {binaryMask.shape}") # (1024, 2048)
        print("OG mask dtype: ", binaryMask.dtype) # uint8
        print(f"Original mask unique values: {np.unique(binaryMask)}") # [ 0  1  3  4  7  8 11 17 20 21 22 23 24 25 26 33]

        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(originalMask, cmap="tab20")
        axes[0].set_title("Máscara etiquetada")
        axes[0].axis("off")
        axes[1].imshow(binaryMask, cmap="gray")
        axes[1].set_title(f"Clase {targetValue} (binaria)")
        axes[1].axis("off")
        plt.tight_layout()
        plt.show()

    try:
        os.makedirs(out_dir, exist_ok=True)
        Image.fromarray(binaryMask).save(os.path.join(out_dir, out_name + ".png"), format='PNG')

    except Exception as e:
        print(f"Error saving file: {e}")
        return



def test_mask_image():
    img_path = dataConfig.CITYSCAPES_ROOT_DIR + "CS_trainLabels/aachen_000000_000019_gtFine_labelIds.png"
    out_dir = dataConfig.CITYSCAPES_ROOT_DIR + "binMasks/train/"
    
    mask_image(img_path, out_dir, "aachen_000000_000019_binMask.png", targetValue = 7)
    print("Masking completed.")

def batch_binMask(originalMaskDir, out_dir, targetValue = 7, filter = r'^(?P<fileId>.*)_gtFine_labelIds.png$'):
    """
    Takes a directory of multiclass label_id mask images and applies the mask_image.

    :param originalMaskDir: Directory containing the input images (multiclass masks).
    :param out_dir: Directory where the binary masks will be saved.
    :param targetValue: The class value to create a binary mask for (default is 7, which is 'road' in Cityscapes).
    :param filter: Regex filter to match files in the originalMaskDir. If None, processes all files.
    """
    if not os.path.exists(originalMaskDir):
        raise FileNotFoundError(f"The directory {originalMaskDir} does not exist.")
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if filter:
        filter = re.compile(filter) # Compile regex filter (raises re.error)

    for dirpath, dirnames, fileNames in os.walk(originalMaskDir):
        print("hola") #debug

        #Use regex filter
        if filter:
            for fileName in fileNames:
                    m = filter.match(fileName)
                    if m:
                        fileId = m.group('fileId') # Save fileId eg: aachen_000000_000019
                        img_path = os.path.join(dirpath, fileName) # path to original mask image
                        out_name = fileId + "_binMask" # name without extension, path will be out_dir + out_name + ".png"

                        print("Masking file " + fileId)
                        mask_image(img_path, out_dir, out_name, targetValue)
                    else: 
                        print(f"File {fileName} does not match the filter. Skipping.")
        
        # No filter -> process all files
        else:
            for fileName in fileNames:
                img_path = os.path.join(dirpath, fileName)
                out_name = "binMask_"+fileName
                
                print("Masking file " + fileName)
                mask_image(img_path, out_dir, out_name, targetValue)

if __name__ == '__main__':
    
    # test_mask_image()

    # make Binary Masks for Val Labels
    def binMaskValLabels():
        originalMasksDir = dataConfig.CITYSCAPES_ROOT_DIR + "CS_valLabels/"
        out_dir = dataConfig.DATA_ROOT_DIR + "binMasks/val/"
        batch_binMask(originalMasksDir, out_dir)

    def binMaskTrainLabels():
        originalMasksDir = dataConfig.CITYSCAPES_ROOT_DIR + "CS_trainLabels/"
        out_dir = dataConfig.DATA_ROOT_DIR + "binMasks/train/"
        batch_binMask(originalMasksDir, out_dir)

    binMaskValLabels()
    binMaskTrainLabels()