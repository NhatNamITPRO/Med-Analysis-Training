import os
import random
import pickle
from statistics import mean

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import monai

from utils import file, image
from dataset import SegformerDataset


# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Training function for one epoch
def train_one_epoch(model, dataloader, criterion, optimizer, device, height, width):
    train_loss_list = []
    model.train()
    tqdm_bar = tqdm(dataloader, total=len(dataloader))

    for batch in tqdm_bar:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits
        upsampled_logits = nn.functional.interpolate(logits, size=(height, width))

        # One-hot encode labels and calculate loss
        one_hot_labels = nn.functional.one_hot(labels.long(), num_classes=2).permute(0, 3, 1, 2)
        loss = criterion(upsampled_logits, one_hot_labels)
        train_loss_list.append(loss.item())

        # Backward pass and optimizer step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Update progress bar
        tqdm_bar.set_description(f"Training Loss: {loss.item():.5f}")

        # Free memory
        del pixel_values, labels, outputs, logits, loss, upsampled_logits, one_hot_labels

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
    train_dataset = SegformerDataset(img_paths=train_img_paths, mask_paths=train_mask_paths, image_processor=image_processor, augment=augment)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Device and model setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    model.decode_head.classifier = nn.Conv2d(768, 2, 1)
    model = model.to(device)
    model = nn.DataParallel(model)
    # Optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")

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
        train_loss_list = train_one_epoch(model, train_dataloader, criterion, optimizer, device, height, width)
        losses.extend(train_loss_list)
        print(f"Mean loss: {mean(train_loss_list):.5f}")

        # Save checkpoint
        ckpt_file_name = os.path.join(output_dir, f"Segformer_epoch_{epoch + 1}.pth")
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
