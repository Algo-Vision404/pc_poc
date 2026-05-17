import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from per_camera_poc import BigPerCameraNet, ARTIFACTS, IMG_SIZE, generate_synthetic_dataset, PerCameraDataset

def unnormalize(tensor):
    return tensor * 0.5 + 0.5

def run_inference_video(model, dataset, device, output_video_path, fps=5):
    """
    Simulates a camera feed by running inference on a sequence of frames
    and writing the overlaid result to an MP4 video.
    """
    print(f"Starting inference on {len(dataset)} frames...")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Video layout: 2x2 grid -> (IMG_SIZE*2, IMG_SIZE*2)
    # 0: Input+Bbox, 1: Depth
    # 2: GT Seg,     3: Pred Seg
    
    out_size = (IMG_SIZE * 2, IMG_SIZE * 2)
    out_video = cv2.VideoWriter(str(output_video_path), fourcc, fps, out_size)
    
    for i in range(len(dataset)):
        sample = dataset[i]
        img_tensor = sample["image"].unsqueeze(0).to(device)
        gt_seg = sample["seg"].numpy()
        
        with torch.no_grad():
            out = model(img_tensor)
            seg_logits = F.interpolate(out["seg"], size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
            pred_seg = torch.argmax(seg_logits, dim=1).squeeze(0).cpu().numpy()
            pred_depth = F.interpolate(out["depth"], size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False).squeeze(0).squeeze(0).cpu().numpy()
            pred_bbox = out["det"][:, 1:5].squeeze(0).cpu().numpy()

        # Input image
        disp_img = unnormalize(sample["image"]).permute(1, 2, 0).numpy()
        disp_img = np.clip(disp_img, 0, 1)
        disp_img = (disp_img * 255).astype(np.uint8)
        disp_img = cv2.cvtColor(disp_img, cv2.COLOR_RGB2BGR)
        
        # Draw Bounding Box (Red)
        pred_xc, pred_yc, pred_w, pred_h = pred_bbox
        x1 = int((pred_xc - pred_w/2) * IMG_SIZE)
        y1 = int((pred_yc - pred_h/2) * IMG_SIZE)
        x2 = int((pred_xc + pred_w/2) * IMG_SIZE)
        y2 = int((pred_yc + pred_h/2) * IMG_SIZE)
        cv2.rectangle(disp_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(disp_img, "Det", (x1, max(y1-5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)

        # Depth mapping (normalize to 0-255 colormap)
        depth_norm = np.clip(pred_depth, 0, 1)
        depth_color = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)

        # Segmentation
        gt_seg_color = (gt_seg * 255).astype(np.uint8)
        gt_seg_color = cv2.cvtColor(gt_seg_color, cv2.COLOR_GRAY2BGR)
        gt_seg_color[:,:,0] = 0 # tint green
        gt_seg_color[:,:,2] = 0
        
        pred_seg_color = (pred_seg * 255).astype(np.uint8)
        pred_seg_color = cv2.cvtColor(pred_seg_color, cv2.COLOR_GRAY2BGR)
        pred_seg_color[:,:,0] = 0 # tint red
        pred_seg_color[:,:,1] = 0
        
        # Combine into 2x2 grid
        top_row = np.hstack((disp_img, depth_color))
        bottom_row = np.hstack((gt_seg_color, pred_seg_color))
        grid = np.vstack((top_row, bottom_row))
        
        # Add labels
        cv2.putText(grid, "Input + BBox", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(grid, "Depth Map", (IMG_SIZE + 5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(grid, "GT Segmentation", (5, IMG_SIZE + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(grid, "Pred Segmentation", (IMG_SIZE + 5, IMG_SIZE + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        out_video.write(grid)

    out_video.release()
    print(f"Video saved to {output_video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=60, help="Number of frames for synthetic video stream")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = ARTIFACTS / "models" / "poc_checkpoint.pt"
    
    if not model_path.exists():
        print("Checkpoint not found. Run training first.")
        exit(1)

    print("Loading model...")
    model = BigPerCameraNet().to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print("Generating simulation sequence...")
    manifest = generate_synthetic_dataset(args.frames)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    dataset = PerCameraDataset(manifest, transform=transform, max_samples=args.frames)

    output_path = ARTIFACTS / "results" / "inference_stream.mp4"
    run_inference_video(model, dataset, device, output_path, fps=args.fps)
