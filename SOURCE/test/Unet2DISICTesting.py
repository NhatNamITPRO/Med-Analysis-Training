import torch
import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import albumentations as A
from utils import image
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms


def predict_image(model, url, device, transform_img):
    img = image.load_image(url)
    H = img.shape[0]
    W = img.shape[1]
    img__tensor = torch.tensor(img).permute(2,0,1)
    input = transform_img(img__tensor).unsqueeze(0).to(device)  
    model.eval()
    with torch.no_grad():
        output = model(input)
        classes = torch.round(output)
    input = torch.nn.functional.interpolate(input, size=(H, W))*255
    classes = torch.nn.functional.interpolate(classes, size=(H, W))
    return input.squeeze().permute(1,2,0).cpu().numpy(),classes.squeeze().cpu().numpy()

def overlay_mask(raw_image, mask):
    raw_image = np.array(raw_image)
    mask_colored = np.stack([mask * 0, mask * 0, mask * 255], axis=-1).astype(np.uint8)  # Purple overlay
    overlay = 0.7 * raw_image + 0.3 * mask_colored
    overlay = overlay.astype(np.uint8)
    return Image.fromarray(overlay)

def predict(image_path,checkpoint_path,output_dir,height,width):
    transform_img = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((height, width)),
        transforms.ToTensor(),
    ])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
        in_channels=3, out_channels=1, init_features=32, pretrained=True).to(device)
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load(checkpoint_path,weights_only=True,map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'],strict=False)
    image,classes = predict_image(model, image_path, device, transform_img)
    # Hiển thị ảnh gốc và mask
    overlay_image = overlay_mask(image,classes)
    overlay_output_path = f"{output_dir}/overlay.png"
    overlay_image.save(overlay_output_path)
    print(f"Overlay image saved to {overlay_output_path}")
