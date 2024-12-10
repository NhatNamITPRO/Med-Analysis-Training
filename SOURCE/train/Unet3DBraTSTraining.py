import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import monai
from statistics import mean
import pickle
from utils import file, image
from models import UNet3D
from dataset import Brats2021Task1Dataset


# --------------------------- Helper Functions ---------------------------
def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    train_loss_list = []
    model.train()
    tqdm_bar = tqdm(dataloader, total=len(dataloader))
    for item in tqdm_bar:
        print("qq")
        inputs,targets = item[0],item[1]
        inputs, targets = inputs.to(device).float(), targets.to(device).int()
        logits = model(inputs)
        probabilities = torch.sigmoid(logits)
        loss = criterion(probabilities, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss_list.append(loss.detach().item())
        tqdm_bar.set_description(desc=f"Training Loss: {loss.detach().item():.5f}")
        del inputs,targets,logits, loss,probabilities
        torch.cuda.empty_cache()
    return train_loss_list


# --------------------------- Main Training Script ---------------------------
def train(DATA_ROOT_PATH, CHECKPOINT_PATH, OUTPUT_PATH, NUM_EPOCHS, BATCH_SIZE, LR, D, H, W):

    # Set seed for reproducibility
    set_seed(42)

    # Dataset and DataLoader
    case_names = next(os.walk(DATA_ROOT_PATH), (None, None, []))[1]
    train_cases, _ = train_test_split(case_names, test_size=0.2, random_state=42)
    
    train_dataset = Brats2021Task1Dataset(DATA_ROOT_PATH, train_cases, D, H, W)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model, Loss, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D(in_channels=4, num_classes=3).to(device) 
    model = torch.nn.DataParallel(model)
    
    optimizer = AdamW(model.parameters(), lr=LR)
    criterion = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")

    # Load checkpoint if available
    start_epoch, losses = 0, []
    if CHECKPOINT_PATH:
        checkpoint = torch.load(CHECKPOINT_PATH,weights_only=True, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        losses = checkpoint["losses"]

    # Training loop
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"---------- Epoch {epoch + 1} ----------")

        # Train
        train_loss_list = train_one_epoch(model, train_dataloader, criterion, optimizer, device)
        losses.extend(train_loss_list)

        print(f"Mean Training Loss: {mean(train_loss_list):.5f}")

        # Save checkpoint
        checkpoint_file = os.path.join(OUTPUT_PATH, f"Unet3D_Brats2021_epoch_{epoch + 1}_model.pth")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "losses": losses,
        }, checkpoint_file)

        # Plot loss
        image.plot_loss(losses,OUTPUT_PATH)

    # Save final loss
    with open(os.path.join(OUTPUT_PATH, "losses.pkl"), "wb") as file:
        pickle.dump(losses, file)
    print("Training Finished!")


