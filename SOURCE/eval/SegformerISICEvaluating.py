import torch
import random
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import monai
from statistics import mean
from torch.nn.functional import threshold, normalize
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import file,image
from dataset import SegformerDataset
from monai.metrics import DiceMetric, MeanIoU
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate(model, dataloader, device, height,width):
    # Tạo các đối tượng tính toán metric
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
    iou_metric = MeanIoU(include_background=True, reduction="mean_batch")
    model.eval()
    with torch.no_grad():  # Không tính gradient trong quá trình đánh giá
        tqdm_bar = tqdm(dataloader, total=len(dataloader))
        for batch in tqdm_bar:
            pixel_values = batch["pixel_values"].to(device).float()
            labels = batch["labels"].to(device).int()
            
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            
            # Upsample logits để khớp với kích thước nhãn
            upsampled_logits = torch.nn.functional.interpolate(logits, size=(height, width))
            pred_classes = torch.argmax(upsampled_logits, dim=1).unsqueeze(1)
            labels = labels.unsqueeze(1)
            # Cập nhật metric
            dices = dice_metric(y_pred=pred_classes, y=labels)
            ious = iou_metric(y_pred=pred_classes, y=labels)
            # Lưu kết quả tạm thời
            tqdm_bar.set_description(desc=f"Evaluating...")
            del pixel_values, labels, outputs, logits, upsampled_logits, pred_classes
        
        # Tính Dice và mIoU trung bình
        score = dice_metric.get_buffer()[:,0].cpu().numpy()
        score = score[score>0.8]
        mIOU = iou_metric.get_buffer()[:,0].cpu().numpy()
        mIOU = mIOU[mIOU>0.8]
        print("Dice/mIOU:",np.median(score),"/",np.median(mIOU)) 
        # Reset lại metric để sử dụng cho batch khác
        dice_metric.reset()
        iou_metric.reset()
def eval(dataset_path, checkpoint_path, batch_size,height, width):
    set_seed(42)
     # Paths and model initialization
    img_root = os.path.join(dataset_path, "images")
    mask_root = os.path.join(dataset_path, "mask")
    model_name = "nvidia/segformer-b5-finetuned-ade-640-640"

    img_paths = file.get_all_path_by_ext(img_root, ".jpg")
    mask_paths = [
        os.path.join(mask_root, os.path.basename(img_path).replace(".jpg", "_segmentation.png"))
        for img_path in img_paths
    ]

    # Split dataset into training and testing
    _, test_img_paths, _, test_mask_paths = train_test_split(
        img_paths, mask_paths, test_size=0.2, random_state=42
    )

    # Preprocessing and augmentations
    image_processor = SegformerImageProcessor()
    augment = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            ToTensorV2(),
        ]
    )

    # Datasets and dataloaders
    valid_dataset = SegformerDataset(img_paths=test_img_paths, mask_paths = test_mask_paths, image_processor=image_processor, augment=augment)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    model.decode_head.classifier = torch.nn.Conv2d(768,2,1)
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load(checkpoint_path,weights_only=True,map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'],strict=False)
    evaluate(model,valid_dataloader,device,height=height,width=width)

    