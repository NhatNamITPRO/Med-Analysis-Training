import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from monai.metrics import DiceMetric, MeanIoU
from dataset import Brats2021Task1Dataset
from models import UNet3D


# Thiết lập hạt giống ngẫu nhiên
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Hàm đánh giá mô hình
def evaluate(model, dataloader, device):
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch",get_not_nans=False,ignore_empty=True)
    iou_metric = MeanIoU(include_background=True, reduction="mean_batch",get_not_nans=False,ignore_empty=True)
    model.eval()
    with torch.no_grad():
        tqdm_bar = tqdm(dataloader, total=len(dataloader))
        for item in tqdm_bar:
            inputs, targets = item[0].to(device).float(), item[1].to(device).int()
            logits = model(inputs)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > 0.5).int()

            # Tính Dice và mIoU cho batch
            dices = dice_metric(predictions, targets)
            ious = iou_metric(predictions, targets)
         
            tqdm_bar.set_description(desc=f"Evaluating...")
            # Dọn dẹp bộ nhớ
            del inputs, targets, logits, probabilities,predictions
            torch.cuda.empty_cache()

        # Tính Dice và mIoU cho từng lớp
        metrics = {}
        for i, label in enumerate(["TC", "WT", "ET"]):
            dices = dice_metric.get_buffer()[:, i].cpu().numpy()
            ious = iou_metric.get_buffer()[:, i].cpu().numpy()

            dices = dices[dices > 0.2]
            ious = ious[ious > 0.2] 
            metrics[label] = {"Dice": np.median(dices), "mIoU": np.median(ious)}

        # Hiển thị kết quả
        for label, values in metrics.items():
            print(f"{label}: Dice = {values['Dice']:.4f}, mIoU = {values['mIoU']:.4f}")

        # Reset metric cho lần sử dụng sau
        dice_metric.reset()
        iou_metric.reset()


def eval(DATA_ROOT_PATH, CHECKPOINT_PATH, BATCH_SIZE, D, H, W):
    set_seed(42)

    # Chuẩn bị dữ liệu
    case_names = next(os.walk(DATA_ROOT_PATH), (None, None, []))[1]
    _, val_case_names = train_test_split(case_names, test_size=0.2, random_state=42)
    val_dataset = Brats2021Task1Dataset(root_dir=DATA_ROOT_PATH, case_name=val_case_names, D=D, H=H, W=W)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Thiết lập thiết bị và mô hình
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D(in_channels=4, num_classes=3).to(device) 
    model = torch.nn.DataParallel(model)

    # Load checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH,weights_only=True, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # Đánh giá mô hình
    evaluate(model, val_dataloader, device)

