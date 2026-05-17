import argparse
import json
import csv
import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# --- Paths ---
ARTIFACTS = Path("./artifacts").resolve()
DATA_DIR = (ARTIFACTS / "synthetic_data").resolve()
MODEL_DIR = (ARTIFACTS / "models").resolve()
RESULTS_DIR = (ARTIFACTS / "results").resolve()

for d in [ARTIFACTS, DATA_DIR, MODEL_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 128

# --- Dataset Generation ---
def generate_synthetic_dataset(n_samples: int):
    manifest_rows = []
    for i in range(int(n_samples)):
        img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (30, 30, 30))
        seg = Image.new("L", (IMG_SIZE, IMG_SIZE), 0)
        depth = Image.new("F", (IMG_SIZE, IMG_SIZE), 0.0)

        draw = ImageDraw.Draw(img)
        draw_seg = ImageDraw.Draw(seg)
        draw_depth = ImageDraw.Draw(depth)

        w, h = random.randint(12, 40), random.randint(12, 40)
        x, y = random.randint(0, IMG_SIZE - w - 1), random.randint(0, IMG_SIZE - h - 1)
        color = tuple(int(random.uniform(60, 255)) for _ in range(3))

        # Make depth correlate with object size (simulate perspective)
        area = w * h
        min_area = 12 * 12
        max_area = 40 * 40
        area_norm = (area - min_area) / (max_area - min_area + 1e-5)
        dval = 0.1 + 0.8 * area_norm

        draw.rectangle([x, y, x + w, y + h], fill=color)
        draw_seg.rectangle([x, y, x + w, y + h], fill=1)
        for yy in range(y, y + h + 1):
            draw_depth.line([(x, yy), (x + w, yy)], fill=dval)

        xc, yc = (x + w / 2) / IMG_SIZE, (y + h / 2) / IMG_SIZE
        w_norm, h_norm = w / IMG_SIZE, h / IMG_SIZE

        img_path = (DATA_DIR / f"img_{i:04d}.png").resolve()
        seg_path = (DATA_DIR / f"seg_{i:04d}.png").resolve()
        depth_path = (DATA_DIR / f"depth_{i:04d}.npy").resolve()

        img.save(str(img_path))
        seg.save(str(seg_path))
        np.save(str(depth_path), np.array(depth, dtype=np.float32))

        manifest_rows.append({
            "image": str(img_path),
            "seg": str(seg_path),
            "depth": str(depth_path),
            "class": 1,
            "xc": xc, "yc": yc, "w": w_norm, "h": h_norm
        })

    manifest_csv = (ARTIFACTS / "dataset_manifest.csv").resolve()
    with open(manifest_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image","seg","depth","class","xc","yc","w","h"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    return manifest_csv

# --- Dataset Class ---
class PerCameraDataset(Dataset):
    def __init__(self, manifest_csv, transform=None, max_samples=None):
        with open(manifest_csv, "r") as f:
            rows = list(csv.DictReader(f))
        if max_samples:
            rows = rows[:int(max_samples)]
        self.rows = rows
        self.transform = transform or self._default_transform

    def _default_transform(self, img: Image.Image):
        arr = np.asarray(img, dtype=np.float32)/255.0
        arr = np.transpose(arr, (2,0,1))
        return torch.from_numpy(arr).float()

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[int(idx)]
        img = Image.open(r["image"]).convert("RGB")
        seg = Image.open(r["seg"]).convert("L")
        depth = np.load(r["depth"])

        img_t = self.transform(img)
        seg_t = torch.from_numpy(np.array(seg,dtype=np.uint8)).long()
        depth_t = torch.from_numpy(depth.astype(np.float32)).unsqueeze(0).float()
        ann_class = torch.tensor(int(r["class"]),dtype=torch.long)
        bbox = torch.tensor([float(r["xc"]), float(r["yc"]), float(r["w"]), float(r["h"])],dtype=torch.float)

        return {"image": img_t, "seg": seg_t, "depth": depth_t,
                "ann": {"class": ann_class, "bbox": bbox}}

# --- Model (Upgraded Big System) ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class BigPerCameraNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        
        # Segmentation Head
        self.seg = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 2, 1)
        )
        
        # Depth Head
        self.depth = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
        
        # Detection Head
        self.det_features = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(), # 8x8
            nn.Conv2d(64, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(), # 4x4
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.3)
        )
        self.det_cls = nn.Linear(256, 1)
        self.det_bbox = nn.Sequential(nn.Linear(256, 4), nn.Sigmoid())

    def _make_layer(self, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        f = self.layer2(x)
        
        seg_out = self.seg(f)
        depth_out = self.depth(f)
        
        det_feats = self.det_features(f)
        cls_out = self.det_cls(det_feats)
        bbox_out = self.det_bbox(det_feats)
        det_out = torch.cat([cls_out, bbox_out], dim=1)
        
        return {"seg": seg_out, "depth": depth_out, "det": det_out}

# --- Evaluation Metrics ---
def compute_iou(pred_box, gt_box):
    # Format: [xc, yc, w, h]
    pred_x1 = pred_box[:, 0] - pred_box[:, 2]/2
    pred_y1 = pred_box[:, 1] - pred_box[:, 3]/2
    pred_x2 = pred_box[:, 0] + pred_box[:, 2]/2
    pred_y2 = pred_box[:, 1] + pred_box[:, 3]/2
    
    gt_x1 = gt_box[:, 0] - gt_box[:, 2]/2
    gt_y1 = gt_box[:, 1] - gt_box[:, 3]/2
    gt_x2 = gt_box[:, 0] + gt_box[:, 2]/2
    gt_y2 = gt_box[:, 1] + gt_box[:, 3]/2
    
    inter_x1 = torch.max(pred_x1, gt_x1)
    inter_y1 = torch.max(pred_y1, gt_y1)
    inter_x2 = torch.min(pred_x2, gt_x2)
    inter_y2 = torch.min(pred_y2, gt_y2)
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    union_area = pred_area + gt_area - inter_area
    
    iou = inter_area / torch.clamp(union_area, min=1e-6)
    return iou.mean().item()

def evaluate(model, loader, device):
    model.eval()
    total_iou_seg = 0
    total_rmse = 0
    total_iou_det = 0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            imgs, seg_t, depth_t = batch["image"].to(device), batch["seg"].to(device), batch["depth"].to(device)
            bbox_t = batch["ann"]["bbox"].to(device)
            out = model(imgs)
            
            # Seg mIoU
            seg_logits = nn.functional.interpolate(out["seg"], size=seg_t.shape[-2:], mode="bilinear")
            preds_seg = torch.argmax(seg_logits, dim=1)
            intersection = torch.logical_and(preds_seg, seg_t).float().sum()
            union = torch.logical_or(preds_seg, seg_t).float().sum()
            iou_seg = intersection / (union + 1e-6)
            total_iou_seg += iou_seg.item()
            
            # Depth RMSE
            depth_pred = nn.functional.interpolate(out["depth"], size=depth_t.shape[-2:], mode="bilinear")
            rmse = torch.sqrt(nn.functional.mse_loss(depth_pred, depth_t))
            total_rmse += rmse.item()
            
            # Det AP/IoU
            total_iou_det += compute_iou(out["det"][:, 1:5], bbox_t)
            batches += 1
            
    return total_iou_seg/batches, total_rmse/batches, total_iou_det/batches

# --- Collate ---
def collate_fn(batch):
    imgs = torch.stack([b["image"] for b in batch])
    segs = torch.stack([b["seg"] for b in batch])
    depths = torch.stack([b["depth"] for b in batch])
    classes = torch.stack([b["ann"]["class"] for b in batch])
    bboxes = torch.stack([b["ann"]["bbox"] for b in batch])
    return {"image": imgs, "seg": segs, "depth": depths, "ann":{"class": classes,"bbox": bboxes}}

# --- Train ---
def train(manifest_csv, epochs=1, batch_size=4, max_samples=64):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    ds = PerCameraDataset(manifest_csv, transform=transform, max_samples=max_samples)
    
    # Split train/val
    train_size = int(0.8 * len(ds))
    val_size = len(ds) - train_size
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BigPerCameraNet().to(device)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)

    seg_loss_fn = nn.CrossEntropyLoss()
    depth_loss_fn = nn.MSELoss()
    det_cls_loss_fn = nn.BCEWithLogitsLoss()
    det_bbox_loss_fn = nn.SmoothL1Loss()

    best_val_loss = float('inf')
    best_metrics = {}
    
    for ep in range(int(epochs)):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            imgs, seg_t, depth_t = batch["image"].to(device), batch["seg"].to(device), batch["depth"].to(device)
            cls_t = batch["ann"]["class"].to(device).float().unsqueeze(1)
            bbox_t = batch["ann"]["bbox"].to(device)

            opt.zero_grad()
            out = model(imgs)

            seg_logits = nn.functional.interpolate(out["seg"], size=seg_t.shape[-2:], mode="bilinear")
            depth_pred = nn.functional.interpolate(out["depth"], size=depth_t.shape[-2:], mode="bilinear")

            seg_loss = seg_loss_fn(seg_logits, seg_t)
            depth_loss = depth_loss_fn(depth_pred, depth_t)
            det_cls_loss = det_cls_loss_fn(out["det"][:,0:1], cls_t)
            det_bbox_loss = det_bbox_loss_fn(out["det"][:,1:5], bbox_t)

            loss = seg_loss + 30.0 * depth_loss + det_cls_loss + 2.0*det_bbox_loss
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            
        scheduler.step()
        
        # Validation
        if len(val_loader) > 0:
            val_mIoU, val_rmse, val_det_iou = evaluate(model, val_loader, device)
        else:
            val_mIoU, val_rmse, val_det_iou = 0, 0, 0
        avg_train_loss = epoch_loss/len(train_loader)
        
        print(f"Epoch {ep+1}/{epochs} - Loss: {avg_train_loss:.4f} - Val mIoU: {val_mIoU:.4f} - Val RMSE: {val_rmse:.4f} - Val Det IoU: {val_det_iou:.4f}")
        
        if avg_train_loss < best_val_loss:
            best_val_loss = avg_train_loss
            ckpt_p = (MODEL_DIR / "poc_checkpoint.pt").resolve()
            torch.save({"model_state": model.state_dict(), "opt_state": opt.state_dict()}, str(ckpt_p))
            best_metrics = {"val_mIoU": val_mIoU, "val_rmse": val_rmse, "val_det_iou": val_det_iou, "loss": avg_train_loss}

    results = {"ckpt": str(ckpt_p), "best_metrics": best_metrics}
    with open(RESULTS_DIR / "poc_results.json","w") as f:
        json.dump(results,f,indent=2)
        
    # --- Export to ONNX ---
    import sys, io
    print("Exporting model to ONNX format...")
    model.load_state_dict(torch.load(ckpt_p, weights_only=True)["model_state"])
    model.eval()
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    onnx_path = (MODEL_DIR / "poc_model.onnx").resolve()
    try:
        # Redirect stdout to a UTF-8 safe stream to avoid Windows cp1252 crash from PyTorch emoji logs
        _orig_stdout = sys.stdout
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['seg', 'depth', 'det'],
        )
        sys.stdout = _orig_stdout
        print(f"ONNX model saved to {onnx_path}")
        results["onnx_path"] = str(onnx_path)
    except Exception as e:
        sys.stdout = _orig_stdout
        print(f"[WARNING] ONNX export failed (non-critical): {e}")
        print("Checkpoint is still valid for the API backend.")

    return results

# --- Main ---
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    manifest = generate_synthetic_dataset(args.samples)
    res = train(manifest, epochs=args.epochs, batch_size=args.batch, max_samples=args.max_samples)
    print("Training complete. Best metrics:")
    print(json.dumps(res, indent=2))
