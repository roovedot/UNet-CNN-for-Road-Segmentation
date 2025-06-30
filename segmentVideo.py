import torch
from UNet_model import UNet
import albumentations as A
from albumentations.pytorch import ToTensorV2 
import cv2
import numpy as np
from tqdm import tqdm
import os
import unetConfig

############################################################################################

# DEFINE VARIABLES:
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_VID_PATH = "/mnt/01DB23CE72C96D00/Pr.Inv.JavierFernandez/InvProject/testvideo.ts"
OUTPUT_DIR = unetConfig.DATA_ROOT_DIR+"results/" # Must end in /

############################################################################################

# LOAD MODEL:
model = UNet(in_channels=3, out_channels=1).to(DEVICE)
checkpoint = torch.load("my_checkpoint.pth.tar", map_location=DEVICE, weights_only=True)
model.load_state_dict(checkpoint["state_dict"])
model.eval() # evaluation mode


# PRE-PROCESSING:
IMG_HEIGHT, IMG_WIDTH = 256, 512

preprocess = A.Compose([
    A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
    A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
    ToTensorV2()
])


# OPEN VIDEO AND PROCESS FRAME BY FRAME

def overlay_mask(frame: np.ndarray, mask: np.ndarray, alpha: float = 0.4):
    """ 
    Draw mask over the image as a red overlay 

    :param frame: serves as a reference for the shape of the image
    :param mask: binary mask (output of the model, which we want to visualize)
    :param alpha: opacity (less alpha -> more transparent)
    """
    color_mask = np.zeros_like(frame) # Create empty mask shape
    color_mask[..., 2] = (mask * 255).astype(np.uint8) # Create red mask overlay
    
    # Blend the overlay mask with the image
    return cv2.addWeighted(
        src1=frame, # Original Image
        alpha=1.0, # Full opacity
        src2=color_mask, # Mask overlay
        beta=alpha, # Opacity of the overlay (40% by default)
        gamma=0 # Post-Blend Brightness bias (reduce if image is saturated)
        )


# LOAD INPUT VIDEO, MANAGE EXTENSIONS AND SET UP output_path

# --- auto‐detect input/output and codec ---
input_path = INPUT_VID_PATH
base, ext = os.path.splitext(input_path.lower())
if ext not in [".mp4", ".ts"]:
    raise ValueError(f"Unsupported extension {ext}")

fileName = os.path.basename(base)

# Map extensions to FOURCC codes
codec = "mp4v"
fourcc = cv2.VideoWriter_fourcc(*codec)
output_path = os.path.join(OUTPUT_DIR, f"{fileName}_overlay{ext}")

temp_mp4 = os.path.join(OUTPUT_DIR, fileName+"_overlay.mp4")

# SET UP CV2:

cap = cv2.VideoCapture(INPUT_VID_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(filename=temp_mp4, fourcc=fourcc, fps=fps, frameSize=(w, h))

# Debug
if not writer.isOpened():
    raise RuntimeError(f"Unable to open VideoWriter with path {temp_mp4}")



# PREDICT ON THE VIDEO AND MASK
frameCounter = 0
with torch.no_grad():
    while True:
        ret, frame = cap.read() # read each frame
        if not ret:
            print("MASKING DONE.")
            break
        
        frameCounter+=1
        print(f"processing frame {frameCounter}")
        
        # preprocess and predict
        aug = preprocess(image=frame)
        x = aug["image"].unsqueeze(0). to(DEVICE)
        preds = model(x) # Output of model
        prob = torch.sigmoid(preds)[0,0].cpu().numpy() # sigmoid activation
        mask = (prob > 0.5).astype(np.uint8) # mask where probability of road is > 50%

        # resize mask & overlay
        mask_full = cv2.resize(src=mask, dsize=(w, h), interpolation=cv2.INTER_NEAREST)
        out_frame = overlay_mask(frame, mask_full, alpha=0.4)

        writer.write(out_frame) #  Construct the output video

# Stop services
cap.release()
writer.release()

'''
# REMUX TO .ts

import subprocess

# final .ts path
final_ts = os.path.join(OUTPUT_DIR, f"{fileName}_overlay.ts")

# fast remux
subprocess.run([
    "ffmpeg", "-y",
    "-i", temp_mp4,
    "-c", "copy",
    final_ts
], check=True)

print(f"Written TS video: {final_ts}")'''