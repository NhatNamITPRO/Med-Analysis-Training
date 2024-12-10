import os
import numpy as np
import torch
import nibabel as nib
from models import UNet3D
from monai.data import MetaTensor
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from monai.transforms import (
    Orientation, 
    EnsureType,
)

def predict_from_folder(model, folder_path, device, D, H, W):

    """
    Dự đoán kết quả segmentation từ một thư mục chứa các file MRI: flair, t1, t1ce, t2.
    
    Args:
        model: Mô hình segmentation đã được load.
        folder_path: Đường dẫn đến thư mục chứa các file MRI.
        device: Thiết bị chạy mô hình ("cuda" hoặc "cpu").
        D, H, W: Kích thước của đầu vào sau khi crop.

    Returns:
        prediction: Mặt nạ segmentation dự đoán (numpy array).
    """
    MRI_TYPE = ["flair", "t1", "t1ce", "t2"]
    def load_nifti(fp):
        """Tải file NIfTI và trả về dữ liệu và affine matrix."""
        nifti_data = nib.load(fp)
        return nifti_data.get_fdata(), nifti_data.affine

    def normalize(x):
        """Chuẩn hóa dữ liệu về khoảng [0, 1]."""
        scaler = MinMaxScaler(feature_range=(0, 1))
        normalized_1D_array = scaler.fit_transform(x.reshape(-1, x.shape[-1]))
        return normalized_1D_array.reshape(x.shape)

    def orient(x, affine):
        """Chuyển hệ tọa độ về chuẩn RAS."""
        meta_tensor = MetaTensor(x=x, affine=affine)
        oriented_tensor = Orientation(axcodes="RAS")(meta_tensor)
        return EnsureType(data_type="numpy", track_meta=False)(oriented_tensor)

    def crop_brats2021_zero_pixels(x):
        """Cắt giảm kích thước về (D, H, W)."""
        H_start = (x.shape[1] - H) // 2
        W_start = (x.shape[2] - W) // 2
        D_start = (x.shape[3] - D) // 2
        return x[:, H_start:H_start + H, W_start:W_start + W, D_start:D_start + D]

    def preprocess_modality(file_path):
        """Tiền xử lý cho từng modality."""
        data, affine = load_nifti(file_path)
        data = normalize(data)
        data = data[np.newaxis, ...]  # (240, 240, 155) -> (1, 240, 240, 155)
        data = orient(data, affine)
        data = crop_brats2021_zero_pixels(data)
        return data
    folder_name = os.path.basename(folder_path)
    # Tiền xử lý cho các modality
    modalities = []
    for mri_type in MRI_TYPE:
        file_path = os.path.join(folder_path, f"{folder_name}_{mri_type}.nii")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} không tồn tại.")
        
        modality = preprocess_modality(file_path)
        modalities.append(modality)
    
    # Gộp các modality thành tensor
    inputs = np.concatenate(modalities, axis=0)  # (4, D, H, W)
    inputs = torch.tensor(inputs).unsqueeze(0).to(device).float()  # Thêm batch dimension
    # Dự đoán với mô hình
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
        probabilities = torch.sigmoid(logits)
        prediction = (probabilities > 0.5).int()
        # Tính toán metric
    return prediction.squeeze(0).cpu().numpy(),inputs.squeeze(0).cpu().numpy()



def overlay_mask(modalities, prediction):
    modalities = modalities*255
    # Giả sử prediction có kích thước (D, H, W, 3) và modalities có kích thước (D, H, W, C)
    D, H, W = modalities.shape[:3]

    # Khởi tạo một mảng để lưu ảnh overlay cuối cùng
    overlay_all_slices = []
    final_masks = []
    flair_slice_colors = []
    for slice_idx in range(D):
        # Lấy modality flair và dự đoán cho slice này
        flair_slice = modalities[slice_idx, :, :, 0]  # (H, W) - Chọn flair modality
        prediction_slice = prediction[slice_idx, :, :, :]  # (H, W, 3)

        # Tách các mask WT, TC, ET
        wt_mask = prediction_slice[:, :, 1]  # Kênh 2: WT
        tc_mask = prediction_slice[:, :, 0]  # Kênh 1: TC
        et_mask = prediction_slice[:, :, 2]  # Kênh 3: ET

        # Chồng các kênh theo thứ tự ET > TC > WT
        final_mask = np.zeros_like(wt_mask)

        final_mask[et_mask > 0] = 3  # U tăng cường (ET)
        final_mask[(tc_mask > 0) & (final_mask == 0)] = 2  # Lõi u (TC)
        final_mask[(wt_mask > 0) & (final_mask == 0)] = 1  # Toàn bộ khối u (WT)
        final_masks.append(final_mask)
        # Chuyển flair_slice thành ảnh màu với 3 kênh
        flair_slice_color = np.stack((flair_slice,) * 3, axis=-1)  # (H, W, 3)
        flair_slice_colors.append(np.copy(flair_slice_color))
        # Overlay các vùng khác nhau bằng màu RGB
        flair_slice_color[final_mask == 1] = [255, 255, 0]    # WT - Đỏ
        flair_slice_color[final_mask == 2] = [0, 255, 255]    # TC - Xanh lá
        flair_slice_color[final_mask == 3] = [255, 0, 255]    # ET - Xanh dương
        
        # Lưu ảnh overlay màu vào mảng kết quả
        overlay_all_slices.append(flair_slice_color)
    return np.stack(overlay_all_slices)
    


# Main
def predict(folder_path, checkpoint_path, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D(in_channels=4, num_classes=3).to(device) 
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Load checkpoint   
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # Predict
    prediction, modalities = predict_from_folder(model, folder_path, device, D=128, H=128, W=128)

    # Tạo overlay
    overlay_all_slices = overlay_mask(np.transpose(modalities, (3, 2, 1, 0)),
                                      np.transpose(prediction, (3, 2, 1, 0)))

    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)

    # Lưu từng lát ảnh dưới dạng file PNG
    for idx, overlay_slice in enumerate(overlay_all_slices):
        output_path = os.path.join(output_dir, f"overlay_slice_{idx:03d}.png")
        plt.imsave(output_path, overlay_slice.astype(np.uint8))

    print(f"Overlay images saved to {output_dir}")
