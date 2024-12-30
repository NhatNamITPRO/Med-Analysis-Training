# Medical Image Segmentation with U-Net and Segformer

## Overview

This repository implements U-Net 2D, U-Net 3D, Segformer, and Segformer 3D models to segment lesions in dermoscopy images and MRI scans.  
These models are trained and evaluated on ISIC2018 and BraTS2021

The project results are as follows:

![alt text](/SOURCE/assets/1.jpg)
![alt text](/SOURCE/assets/2.jpg)
![alt text](/SOURCE/assets/3.jpg)
---

## Environment Setup Guide
This project requires Python and essential packages listed in the `requirements.txt` file. Follow the steps below to set up the environment and install the dependencies.

### 1. Check Python Version
Ensure that Python 3.10 or higher is installed on your system. Check the version using:

```bash
python --version
```

or on some systems:

```bash
python3 --version
```

### 2. Create a Virtual Environment
#### On Windows:
```bash
python -m venv venv
```

#### On Linux/MacOS:
```bash
python3 -m venv venv
```

This will create a `venv` folder containing the virtual environment.

### 3. Activate the Virtual Environment
#### On Windows:
```bash
venv\Scripts\activate
```

#### On Linux/MacOS:
```bash
source venv/bin/activate
```

Once activated, you will see the virtual environment name (e.g., `(venv)`) at the beginning of the command line.

### 4. Install Required Packages
Make sure the `requirements.txt` file is in your working directory. Install the dependencies using:

```bash
pip install -r requirements.txt
```

### 5. Verify Installation
Check that all required packages are installed by running:

```bash
pip freeze
```

Compare the installed packages with those listed in `requirements.txt`.

### 6. Deactivate the Virtual Environment (After Use)
To exit the virtual environment, run:

```bash
deactivate
```

### Notes:
- Use the correct Python version as required by the project.
- If you encounter installation errors, ensure `pip` is up to date:

```bash
pip install --upgrade pip
```

---

## 7. Download Model Checkpoints
Navigate to the `SETUP` folder and run `main.py` to download model checkpoints:

```bash
cd SETUP
python main.py
```

---

## 8. Usage
Move to the `SOURCE` folder and run `main.py` to train, evaluate, or test the models. The results will be displayed in the console or saved in the output folder.

### Example Commands:
#### Training Segformer3D on BraTS2021:
```bash
python main.py --model_name Segformer3D --mode training --dataset_path ./data/brats --model_checkpoint ./checkpoints/Segformer3D_BraTS2021_epoch_50_model.pth --output_dir ./output --epochs 53 --batch_size 1
```

#### Evaluating Segformer3D on BraTS2021:
```bash
python main.py --model_name Segformer3D --mode evaluating --dataset_path ./data/brats --model_checkpoint ./checkpoints/Segformer3D_BraTS2021_epoch_50_model.pth --output_dir ./output --batch_size 1
```

#### Testing Segformer3D on BraTS2021:
```bash
python main.py --model_name Segformer3D --mode testing --input_path ./data/brats/BraTS2021_00003 --model_checkpoint ./checkpoints/Segformer3D_BraTS2021_epoch_50_model.pth --output_dir ./output
```

#### Training U-Net 2D on ISIC2018:
```bash
python main.py --model_name Unet2D --mode training --dataset_path ./data/isic --model_checkpoint ./checkpoints/Unet2D_ISIC2018_epoch_50_model.pth --output_dir ./output --epochs 52 --batch_size 1 --H 256 --W 256
```

#### Testing U-Net 2D on ISIC2018:
```bash
python main.py --model_name Unet2D --mode testing --input_path ./data/isic/images/ISIC_0000000.jpg --model_checkpoint ./checkpoints/Unet2D_ISIC2018_epoch_50_model.pth --output_dir ./output --H 256 --W 256
```

Additional commands for other models and tasks are structured similarly.

---

## 9. App Demo
Navigate to the `app` folder, which contains links to the client, server, and model repositories. You can also access the deployed app at:  
[Medical Analysis App](https://med-analysis-lyart.vercel.app/)

```bash
cd app
```
