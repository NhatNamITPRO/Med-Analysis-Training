import os
import random
import pickle
from statistics import mean

import numpy as np
import torch
from torch.optim import AdamW
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import monai
import warnings
warnings.filterwarnings("ignore")
from utils import file, image
from dataset import UnetDataset

# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Training function for one epoch
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    train_loss_list = []
    model.train()
    tqdm_bar = tqdm(dataloader, total=len(dataloader))
    for X,Y in tqdm_bar:
        X,Y = X.to(device),Y.to(device) #(Y: BHW)
        output = model(X)
        loss = criterion(output, Y.unsqueeze(1))
        # Tính toán loss bằng criterion
        train_loss_list.append(loss.detach().item())
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Cập nhật thanh tiến trình với giá trị loss và metric
        tqdm_bar.set_description(desc=f"Training Loss: {loss.detach().item():.5f}")
        
        # Giải phóng bộ nhớ
        del X, Y, output, loss
        torch.cuda.empty_cache()
    return train_loss_list



# Main training function
def train(dataset_path, checkpoint_path, output_dir, batch_size, epochs, height, width,lr):
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
    train_img_paths, _, train_mask_paths, _ = train_test_split(
        img_paths, mask_paths, test_size=0.2, random_state=42
    )

    # Preprocessing and augmentations
    transform_img = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((height, width)),
        transforms.ToTensor(),
    ])
    transform_mask = transforms.Compose([
        transforms.Resize((height, width),interpolation=transforms.InterpolationMode.NEAREST),
    ])
    aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            ToTensorV2()
        ])

    # Datasets and dataloaders
    train_dataset = UnetDataset(train_img_paths,train_mask_paths,transform_img=transform_img,transform_mask=transform_mask,augment=aug)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
        in_channels=3, out_channels=1, init_features=32, pretrained=True)
    model.to(device)
    model = torch.nn.DataParallel(model)
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    # Load checkpoint if available
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path,map_location=device,weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        losses = checkpoint["losses"]
    else:
        start_epoch = 0
        losses = []

    # Training loop
    for epoch in range(start_epoch, epochs):
        print(f"---------- Epoch {epoch + 1} ----------")
        train_loss_list = train_one_epoch(model, train_dataloader, criterion, optimizer, device)
        losses.extend(train_loss_list)
        print(f"Mean loss: {mean(train_loss_list):.5f}")

        # Save checkpoint
        ckpt_file_name = os.path.join(output_dir, f"UNET2D_ISIC2018_epoch_{epoch + 1}.pth")
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "losses": losses,
            },
            ckpt_file_name,
        )

        # Plot losses
        image.plot_loss(losses, output_dir)

    # Save losses to file
    with open(os.path.join(output_dir, "losses.pkl"), "wb") as f:
        pickle.dump(losses, f)

    print("Training Finished!")
