
# Hướng Dẫn Thiết Lập Môi Trường
Dự án này yêu cầu sử dụng Python và các gói cần thiết được liệt kê trong tệp `requirements.txt`. Dưới đây là các bước để thiết lập môi trường và cài đặt các gói.

## 1. Kiểm tra phiên bản Python

Đảm bảo rằng bạn đã cài đặt Python 3.10 trở lên trên hệ thống của mình.  
Kiểm tra bằng lệnh:

```bash
python --version
```

Hoặc (trên một số hệ thống):

```bash
python3 --version
```

## 2. Tạo môi trường ảo

### Trên Windows:
```bash
python -m venv venv
```

### Trên Linux/MacOS:
```bash
python3 -m venv venv
```

Lệnh trên sẽ tạo một thư mục `env` chứa môi trường ảo.

## 3. Kích hoạt môi trường ảo

### Trên Windows:
```bash
venv\Scripts\activate
```

### Trên Linux/MacOS:
```bash
source venv/bin/activate
```

Sau khi kích hoạt thành công, bạn sẽ thấy tên môi trường ảo xuất hiện ở đầu dòng lệnh, ví dụ: `(env)`.

## 4. Cài đặt các gói

Đảm bảo rằng tệp `requirements.txt` có trong thư mục làm việc của bạn. Chạy lệnh sau để cài đặt các gói:

```bash
pip install -r requirements.txt
```

## 5. Kiểm tra cài đặt

Đảm bảo rằng tất cả các gói đã được cài đặt bằng cách chạy lệnh sau:

```bash
pip freeze
```

Danh sách các gói đã cài đặt sẽ được hiển thị. Hãy kiểm tra xem tất cả các gói trong `requirements.txt` đã được cài đặt.

## 6. Vô hiệu hóa môi trường ảo (Khi hoàn tất)

Khi không còn sử dụng, bạn có thể thoát khỏi môi trường ảo bằng lệnh:

```bash
deactivate
```

---

### Lưu ý

- Hãy đảm bảo sử dụng đúng phiên bản Python mà dự án yêu cầu.
- Nếu gặp lỗi trong quá trình cài đặt, hãy kiểm tra lại tệp `requirements.txt` hoặc cập nhật `pip` lên phiên bản mới nhất:

```bash
pip install --upgrade pip
```
Danh sách các gói đã cài đặt sẽ được hiển thị. Hãy kiểm tra xem tất cả các gói trong `requirements.txt` đã được cài đặt.

## 7. Download các model checkpoint

Di chuyển vào thư mục SETUP và chạy file main.py để tiến hành download các model checkpoint
```bash
cd SETUP
python main.py
```

## 8. Bắt đầu sử dụng

Di chuyển ra ngoài và vào thư mục SOURCE và chạy file main.py theo các hướng dẫn dưới đây hoặc tùy chỉnh
Kết quả được hiển thị ở màn hình console hoặc folder output
```bash
cd ../SOURCE
```
Chạy câu lệnh bên dưới để thực hiện training Segformer3D trên tập BraTS2021
```bash
python main.py --model_name Segformer3D --mode training --dataset_path ./data/brats --model_checkpoint ./checkpoints/Segformer3D_BraTS2021_epoch_50_model.pth --output_dir ./output --epochs 53 --batch_size 1
```
Chạy câu lệnh bên dưới để thực hiện evaluating Segformer3D trên tập BraTS2021
```bash
python main.py --model_name Segformer3D --mode evaluating --dataset_path ./data/brats --model_checkpoint ./checkpoints/Segformer3D_BraTS2021_epoch_50_model.pth --output_dir ./output --batch_size 1
```
Chạy câu lệnh bên dưới để thực hiện testing Segformer3D trên tập BraTS2021
```bash
python main.py --model_name Segformer3D --mode testing --input_path ./data/brats/BraTS2021_00003 --model_checkpoint ./checkpoints/Segformer3D_BraTS2021_epoch_50_model.pth --output_dir ./output
```
Chạy câu lệnh bên dưới để thực hiện training Unet3D trên tập BraTS2021
```bash
python main.py --model_name Unet3D --mode training --dataset_path ./data/brats --model_checkpoint ./checkpoints/Unet3D_BraTS2021_epoch_50_model.pth --output_dir ./output --epochs 53 --batch_size 1
```
Chạy câu lệnh bên dưới để thực hiện evaluating Unet3D trên tập BraTS2021
```bash
python main.py --model_name Unet3D --mode evaluating --dataset_path ./data/brats --model_checkpoint ./checkpoints/Unet3D_BraTS2021_epoch_50_model.pth --output_dir ./output --batch_size 1
```
Chạy câu lệnh bên dưới để thực hiện testing Unet3D trên tập BraTS2021
```bash
python main.py --model_name Unet3D --mode testing --input_path ./data/brats/BraTS2021_00003 --model_checkpoint ./checkpoints/Unet3D_BraTS2021_epoch_50_model.pth --output_dir ./output
```
Chạy câu lệnh bên dưới để thực hiện training Segformer trên tập ISIC2018
```bash
python main.py --model_name Segformer --mode training --dataset_path ./data/isic --model_checkpoint ./checkpoints/Segformer_ISIC2018_epoch_50_model.pth --output_dir ./output --epochs 52 --batch_size 1 --H 512 --W 512
```
Chạy câu lệnh bên dưới để thực hiện evaluating Segformer trên tập ISIC2018
```bash
python main.py --model_name Segformer --mode evaluating --dataset_path ./data/isic --model_checkpoint ./checkpoints/Segformer_ISIC2018_epoch_50_model.pth --output_dir ./output --batch_size 1 --H 512 --W 512
```
Chạy câu lệnh bên dưới để thực hiện testing Segformer trên tập ISIC2018
```bash
python main.py --model_name Segformer --mode testing --input_path ./data/isic/images/ISIC_0000000.jpg --model_checkpoint ./checkpoints/Segformer_ISIC2018_epoch_50_model.pth --output_dir ./output
```
Chạy câu lệnh bên dưới để thực hiện training Unet2D trên tập ISIC2018
```bash
python main.py --model_name Unet2D --mode training --dataset_path ./data/isic --model_checkpoint ./checkpoints/Unet2D_ISIC2018_epoch_50_model.pth --output_dir ./output --epochs 52 --batch_size 1 --H 256 --W 256
```
Chạy câu lệnh bên dưới để thực hiện evaluating Unet2D trên tập ISIC2018
```bash
python main.py --model_name Unet2D --mode evaluating --dataset_path ./data/isic --model_checkpoint ./checkpoints/Unet2D_ISIC2018_epoch_50_model.pth --output_dir ./output --batch_size 1 --H 256 --W 256
```
Chạy câu lệnh bên dưới để thực hiện testing Unet2D trên tập ISIC2018
```bash
python main.py --model_name Unet2D --mode testing --input_path ./data/isic/images/ISIC_0000000.jpg --model_checkpoint ./checkpoints/Unet2D_ISIC2018_epoch_50_model.pth --output_dir ./output --H 256 --W 256
```
## 9. APP DEMO
Di chuyển vào thư mục app, thư mục app chưa các link github kho lưu trữ source của client, server và các model
Bạn có thể truy cập vào link app đã deploy https://med-analysis-lyart.vercel.app/
```bash
cd app
```