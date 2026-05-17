import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from per_camera_poc import BigPerCameraNet, PerCameraDataset, ARTIFACTS, IMG_SIZE

# --- Paths ---
MODEL_DIR = ARTIFACTS / "models"
RESULTS_DIR = ARTIFACTS / "results"
DATA_DIR = ARTIFACTS / "synthetic_data"

CHECKPOINT_PATH = MODEL_DIR / "poc_checkpoint.pt"

# --- Load checkpoint ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BigPerCameraNet().to(device)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# --- Load dataset (few examples for visualization) ---
manifest_csv = ARTIFACTS / "dataset_manifest.csv"
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
dataset = PerCameraDataset(manifest_csv, transform=transform, max_samples=10)  # visualize 10 samples

def unnormalize(tensor):
    return tensor * 0.5 + 0.5

# --- Visualization ---
for i in range(min(5, len(dataset))):
    sample = dataset[i]
    img = sample["image"].unsqueeze(0).to(device)
    gt_seg = sample["seg"].numpy()
    gt_depth = sample["depth"].squeeze(0).numpy()
    gt_bbox = sample["ann"]["bbox"].numpy()

    with torch.no_grad():
        out = model(img)

        # Resize seg to match ground truth
        seg_logits = F.interpolate(
            out["seg"], size=gt_seg.shape, mode="bilinear", align_corners=False
        )
        pred_seg = torch.argmax(seg_logits, dim=1).squeeze(0).cpu().numpy()

        # Resize depth to match
        pred_depth = F.interpolate(
            out["depth"], size=gt_depth.shape, mode="bilinear", align_corners=False
        ).squeeze(0).squeeze(0).cpu().numpy()
        
        pred_bbox = out["det"][:, 1:5].squeeze(0).cpu().numpy()

    print(f"Sample {i}")
    print(" GT Seg unique:", np.unique(gt_seg))
    print(" Pred Seg unique:", np.unique(pred_seg))

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    disp_img = unnormalize(sample["image"]).permute(1, 2, 0).numpy()
    disp_img = np.clip(disp_img, 0, 1)
    
    axes[0].imshow(disp_img)
    axes[0].set_title("Input Image & Detection")
    axes[0].axis("off")
    
    # Draw GT bounding box (green)
    gt_xc, gt_yc, gt_w, gt_h = gt_bbox
    gt_rect = patches.Rectangle(
        ((gt_xc - gt_w/2)*IMG_SIZE, (gt_yc - gt_h/2)*IMG_SIZE), 
        gt_w*IMG_SIZE, gt_h*IMG_SIZE, 
        linewidth=2, edgecolor='g', facecolor='none', label='GT'
    )
    axes[0].add_patch(gt_rect)
    
    # Draw Pred bounding box (red)
    pred_xc, pred_yc, pred_w, pred_h = pred_bbox
    pred_rect = patches.Rectangle(
        ((pred_xc - pred_w/2)*IMG_SIZE, (pred_yc - pred_h/2)*IMG_SIZE), 
        pred_w*IMG_SIZE, pred_h*IMG_SIZE, 
        linewidth=2, edgecolor='r', facecolor='none', label='Pred', linestyle='--'
    )
    axes[0].add_patch(pred_rect)
    axes[0].legend()

    axes[1].imshow(gt_seg, cmap="gray")
    axes[1].set_title("Ground Truth Segmentation")
    axes[1].axis("off")

    axes[2].imshow(pred_seg, cmap="gray")
    axes[2].set_title("Predicted Segmentation")
    axes[2].axis("off")

    im = axes[3].imshow(pred_depth, cmap="viridis")
    axes[3].set_title("Predicted Depth")
    axes[3].axis("off")
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    
    save_path = RESULTS_DIR / f"sample_{i:03d}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
