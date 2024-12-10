import os
import gdown
if __name__ == "__main__":
    try:
        segformer_checkpoint = "1zZ3XbfixwiY3Tra78EvD5siMJIF6IvBW"
        segformer_output = "../SOURCE/checkpoints/Segformer_ISIC2018_epoch_50_model.pth"
        if not os.path.exists(segformer_output):
            gdown.download(f"https://drive.google.com/uc?id={segformer_checkpoint}&confirm=t&uuid=df1eac8a-fdc0-4438-9a29-202168235570", segformer_output, quiet=False)
        
        segformer3D_checkpoint = "1qtWBuwE8PVb-_dzLbl_ySEPX6fNtEGBS"
        segformer3D_output = "../SOURCE/checkpoints/Segformer3D_BraTS2021_epoch_50_model.pth"
        if not os.path.exists(segformer3D_output):
            gdown.download(f"https://drive.google.com/uc?id={segformer3D_checkpoint}&confirm=t&uuid=df1eac8a-fdc0-4438-9a29-202168235570", segformer3D_output, quiet=False)
        
        unet2D_checkpoint = "1c4QD-enJLe32UJcKINP9pwXSShh6q2YW"
        unet2D_output = "../SOURCE/checkpoints/Unet2D_ISIC2018_epoch_50_model.pth"
        if not os.path.exists(unet2D_output):
            gdown.download(f"https://drive.google.com/uc?id={unet2D_checkpoint}&confirm=t&uuid=df1eac8a-fdc0-4438-9a29-202168235570", unet2D_output, quiet=False)
        unet3D_checkpoint = "14nZHB0yEUXD2zNtcQkU5rwxk_PQbRVLh"
        unet3D_output = "../SOURCE/checkpoints/Unet3D_BraTS2021_epoch_50_model.pth"
        if not os.path.exists(unet3D_output):
            gdown.download(f"https://drive.google.com/uc?id={unet3D_checkpoint}&confirm=t&uuid=df1eac8a-fdc0-4438-9a29-202168235570", unet3D_output, quiet=False)
        print("Checkpoints are downloaded at /SOURCE/checkpoints")
    except Exception as e:
        print(f"Đã xảy ra lỗi khi download: {e}")



