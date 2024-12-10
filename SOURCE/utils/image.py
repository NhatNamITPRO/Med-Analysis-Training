import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
def load_image(img_path):
    """
    Helper function to load an image and convert it to a numpy array.

    Parameters:
        img_path (str): Path to the image file.

    Returns:
        np.ndarray: Loaded image as a numpy array with shape (H, W) for grayscale images
                    or (H, W, C) for color images (e.g., RGB or RGBA).
    """
    img = Image.open(img_path)
    return np.array(img)
def plot_image(image: np.ndarray, title: str = "Image", cmap: str = None):
    """
    Plot an image with an optional title and colormap.

    Parameters:
        image (np.ndarray): The image to plot. Can be 2D (grayscale) or 3D (RGB/RGBA).
        title (str): The title of the plot. Default is "Image".
        cmap (str): Colormap to use if the image is grayscale (2D). Default is None.

    Returns:
        None
    """
    plt.figure(figsize=(6, 6))
    if image.ndim == 2:
        plt.imshow(image, cmap=cmap)
    else:
        plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()

def save_image(image: np.ndarray, save_path: str, create_dir: bool = True):
    """
    Save the image to a specified path.

    Parameters:
        image (np.ndarray): The image to save. Should be in RGB, RGBA, or grayscale format.
        save_path (str): Path to save the image, including the filename and extension.
        create_dir (bool): If True, creates the directory if it doesn't exist. Default is True.
    
    Returns:
        None
    """
    # Tạo thư mục nếu chưa tồn tại (nếu create_dir=True)
    if create_dir:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Kiểm tra nếu đầu vào là numpy array
    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be a numpy array.")

    # Chuyển đổi numpy array thành đối tượng Image
    try:
        img_to_save = Image.fromarray(image)
    except ValueError as e:
        raise ValueError("Failed to convert numpy array to image. Ensure the array format is correct.") from e

    # Lưu ảnh với đường dẫn chỉ định
    try:
        img_to_save.save(save_path)
        print(f"Image successfully saved at: {save_path}")
    except Exception as e:
        raise IOError(f"Failed to save image at {save_path}.") from e
def plot_loss(losses, output_dir="output"):
    """
    Plot the loss over iters and save the plot as a PNG image.

    Parameters:
        losses (list or np.ndarray): List or array of loss values for each iter.
        output_dir (str): Directory to save the plot image. Default is "output".
    
    Returns:
        None
    """
    # Kiểm tra nếu losses không rỗng
    if not losses:
        raise ValueError("The 'losses' list is empty. Cannot plot loss.")

    # Tạo thư mục OUTPUT_DIR nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)

    # Tạo đồ thị
    figure, ax = plt.subplots()
    ax.plot(losses, label='Loss', color='blue')  # Vẽ đồ thị
    ax.set_xlabel('Iter')
    ax.set_ylabel('Loss')
    ax.set_title('Loss per Iter')  # Thêm tiêu đề
    ax.legend()

    # Lưu hình ảnh
    save_path = os.path.join(output_dir, "loss.png")
    figure.savefig(save_path)
    print(f"Loss plot saved at: {save_path}")
    plt.close(figure)  # Đóng figure để giải phóng tài nguyên
def get_bounding_box(ground_truth_map):
    """
    Get bounding box coordinates from a ground truth mask, with random perturbations to the coordinates.
    
    Parameters:
        ground_truth_map (np.ndarray): Binary mask where non-zero values indicate the region of interest.
    
    Returns:
        list: Bounding box coordinates [x_min, y_min, x_max, y_max].
    """
    # Get indices of non-zero values in the ground truth map
    y_indices, x_indices = np.where(ground_truth_map > 0)
    
    if len(x_indices) == 0 or len(y_indices) == 0:
        raise ValueError("The ground truth map does not contain any non-zero values.")
    
    # Calculate min and max coordinates
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    
    # Add perturbation to the bounding box coordinates (with a limit to stay within the image bounds)
    H, W = ground_truth_map.shape
    perturbation = np.random.randint(0, 20)
    
    # Apply perturbation while ensuring the bounding box stays within the image boundaries
    x_min = max(0, x_min - perturbation)
    x_max = min(W, x_max + perturbation)
    y_min = max(0, y_min - perturbation)
    y_max = min(H, y_max + perturbation)
    
    bbox = [x_min, y_min, x_max, y_max]
    return bbox
def show_box(box, ax, edgecolor='green', linewidth=2):
    """
    Draw a bounding box on an image.

    Parameters:
        box (list): Coordinates of the bounding box in the format [x_min, y_min, x_max, y_max].
        ax (matplotlib.axes.Axes): The Axes object where the bounding box will be drawn.
        edgecolor (str): The color of the bounding box edge. Default is 'green'.
        linewidth (int): The thickness of the bounding box edge. Default is 2.
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0,0,0,0), lw=linewidth))

def show_boxes_on_image(raw_image, boxes, title="Image with Boxes"):
    """
    Draw multiple bounding boxes on an image.

    Parameters:
        raw_image (np.ndarray): The image on which the bounding boxes will be drawn.
        boxes (list of list): A list of bounding boxes, each represented by [x_min, y_min, x_max, y_max].
        title (str): The title for the image. Default is "Image with Boxes".
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(raw_image)
    for box in boxes:
        show_box(box, plt.gca())  # Draw each bounding box
    plt.title(title)  # Add a title to the image
    plt.axis('on')  # Display the axes
    plt.show()
def show_mask(image, mask, ax, random_color=False):
    """
    Overlay a mask on top of an image and display it.

    Parameters:
        image (np.ndarray): The image on which to overlay the mask.
        mask (np.ndarray): The mask to overlay on the image. It should be a 2D binary array (or have shape [H, W]).
        ax (matplotlib.axes.Axes): The Axes object where the image and mask will be displayed.
        random_color (bool): If True, assigns a random color to the mask. Default is False (uses a predefined color).
    
    Returns:
        None
    """
    ax.imshow(image)  # Display the image
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)  # Generate a random color
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])  # Predefined color (light blue with transparency)

    h, w = mask.shape[-2:]  # Get the height and width of the mask
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)  # Apply the mask and color
    ax.imshow(mask_image)  # Overlay the mask on top of the image