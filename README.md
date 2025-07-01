# UNet CNN for Road Segmentation

A lightweight PyTorch implementation of the U‑Net architecture for semantic segmentation of road surfaces in dashcam video frames.
  
## 📖 Documentation

For a detailed project overview, architecture diagrams, code explanations, and next steps, please refer to the full write‑up (PDF):

[UNet CNN for Road Segmentation Documentation (Public Notion Page)](https://massive-birch-56b.notion.site/UNet-CNN-for-road-segmentation-1bfd04bb9417808c9a96f42df9f03984?pvs=74) --> (https://massive-birch-56b.notion.site/UNet-CNN-for-road-segmentation-1bfd04bb9417808c9a96f42df9f03984?pvs=74)

## 🛠️ Project Structure

```
├─ ETL/                   # Data extraction & mask generation utilities
├─ dataset.py             # Custom PyTorch Dataset for images & masks
├─ unetUtils.py           # Data loaders, checkpointing, metrics, visualization
├─ UNet_model.py          # U‑Net architecture implementation
├─ train.py               # Hyperparameter and Augmentation config, Training & validation loop
├─ segmentVideo.py        # Inference script, outputs the input video (.mp4 or .ts), + an overlay binary mask (road prediction)
└─ README.md              # This file
```

## 📜 License

This project is licensed under the GPL-3.0 License.
