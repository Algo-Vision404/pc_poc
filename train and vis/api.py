import io
import base64
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms
import sys
import os

# Ensure the module can be found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from per_camera_poc import BigPerCameraNet, ARTIFACTS, IMG_SIZE

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BigPerCameraNet().to(device)

model_path = ARTIFACTS / "models" / "poc_checkpoint.pt"
if model_path.exists():
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def tensor_to_b64(tensor_img, is_mask=False, is_depth=False):
    if is_mask:
        arr = (tensor_img * 255).astype(np.uint8)
        arr = cv2.applyColorMap(arr, cv2.COLORMAP_JET)
    elif is_depth:
        arr = np.clip(tensor_img, 0, 1)
        arr = (arr * 255).astype(np.uint8)
        arr = cv2.applyColorMap(arr, cv2.COLORMAP_VIRIDIS)
    else:
        arr = (tensor_img * 255).astype(np.uint8)
    
    _, buffer = cv2.imencode('.png', arr)
    return base64.b64encode(buffer).decode('utf-8')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    original_size = img.size
    
    img_t = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        out = model(img_t)
        
        seg_logits = F.interpolate(out["seg"], size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        pred_seg = torch.argmax(seg_logits, dim=1).squeeze(0).cpu().numpy()
        
        pred_depth = F.interpolate(out["depth"], size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False).squeeze(0).squeeze(0).cpu().numpy()
        
        pred_bbox = out["det"][:, 1:5].squeeze(0).cpu().numpy()
        cls_score = torch.sigmoid(out["det"][:, 0]).item()
        
    # Convert original to cv2 for base64
    orig_cv2 = cv2.cvtColor(np.array(img.resize((IMG_SIZE, IMG_SIZE))), cv2.COLOR_RGB2BGR)
    
    # Draw bbox on original
    pred_xc, pred_yc, pred_w, pred_h = pred_bbox
    x1 = int((pred_xc - pred_w/2) * IMG_SIZE)
    y1 = int((pred_yc - pred_h/2) * IMG_SIZE)
    x2 = int((pred_xc + pred_w/2) * IMG_SIZE)
    y2 = int((pred_yc + pred_h/2) * IMG_SIZE)
    cv2.rectangle(orig_cv2, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(orig_cv2, f"Det: {cls_score:.2f}", (x1, max(y1-5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)

    return {
        "input_b64": tensor_to_b64(orig_cv2),
        "seg_b64": tensor_to_b64(pred_seg, is_mask=True),
        "depth_b64": tensor_to_b64(pred_depth, is_depth=True),
        "bbox": {"x": float(pred_xc), "y": float(pred_yc), "w": float(pred_w), "h": float(pred_h)},
        "score": float(cls_score)
    }

@app.get("/")
def health_check():
    return {"status": "ok", "model": "BigPerCameraNet"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
