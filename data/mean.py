import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def compute_rgb_mean_std(image_folder):
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    # ----- First pass: compute mean -----
    print("Computing mean...")
    for root, _, files in tqdm(os.walk(image_folder)):
        for filename in tqdm(files,leave=False):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, filename)

                img = Image.open(img_path).convert('RGB')
                img_np = np.array(img, dtype=np.float64)

                pixel_sum += img_np.sum(axis=(0, 1))
                pixel_count += img_np.shape[0] * img_np.shape[1]

    mean_rgb = pixel_sum / pixel_count

    # ----- Second pass: compute std -----
    sq_diff_sum = np.zeros(3, dtype=np.float64)
    print("Computing Std...")
    for root, _, files in tqdm(os.walk(image_folder)):
        for filename in tqdm(files,leave=False):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, filename)

                img = Image.open(img_path).convert('RGB')
                img_np = np.array(img, dtype=np.float64)

                sq_diff_sum += ((img_np - mean_rgb) ** 2).sum(axis=(0, 1))

    std_rgb = np.sqrt(sq_diff_sum / pixel_count)

    return mean_rgb, std_rgb


folder_path = "pop3d-seg"
(mean_r, mean_g, mean_b), (std_r, std_g, std_b) = compute_rgb_mean_std(folder_path)

print(f"Mean R: {mean_r:.2f}, Std R: {std_r:.2f}")
print(f"Mean G: {mean_g:.2f}, Std G: {std_g:.2f}")
print(f"Mean B: {mean_b:.2f}, Std B: {std_b:.2f}")

