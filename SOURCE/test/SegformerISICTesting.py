import torch
import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import albumentations as A
from utils import image
import matplotlib.pyplot as plt
from PIL import Image

def predict_image(inputs, model,H,W):
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits 
    upsampled_logits = torch.nn.functional.interpolate(logits, size=(H, W))
    predictions = torch.argmax(upsampled_logits, dim=1).squeeze().cpu().numpy()  
    
    return predictions

def overlay_mask(raw_image, mask):
    raw_image = np.array(raw_image)
    mask_colored = np.stack([mask * 0, mask * 0, mask * 255], axis=-1).astype(np.uint8)  # Purple overlay
    overlay = 0.7 * raw_image + 0.3 * mask_colored
    overlay = overlay.astype(np.uint8)
    return Image.fromarray(overlay)

def predict(image_path,checkpoint_path,output_dir):
    model_name = "nvidia/segformer-b5-finetuned-ade-640-640"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    model.decode_head.classifier = torch.nn.Conv2d(768,2,1)
    model = model.to(device)
    model = torch.nn.DataParallel(model)

    checkpoint = torch.load(checkpoint_path,weights_only=True,map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'],strict=False)
    model.eval()

    image_processor = SegformerImageProcessor()
    raw_image = image.load_image(image_path)
    inputs = image_processor(images=raw_image, return_tensors="pt").to(device)
    segmentation_mask = predict_image(inputs, model,raw_image.shape[0],raw_image.shape[1])
    # Hiển thị ảnh gốc và mask
    overlay_image = overlay_mask(raw_image,segmentation_mask)
    overlay_output_path = f"{output_dir}/overlay.png"
    overlay_image.save(overlay_output_path)
    print(f"Overlay image saved to {overlay_output_path}")
