import os
import shutil
from tqdm import tqdm

sar_root = "kerala_tiles_output_sentinel"
mask_root = "kerala_processed_masks"

out_img_dir = "dataset/images"
out_mask_dir = "dataset/masks"

os.makedirs(out_img_dir, exist_ok=True)
os.makedirs(out_mask_dir, exist_ok=True)

missing_masks = 0
total = 0

for grid in os.listdir(sar_root):
    grid_path = os.path.join(sar_root, grid)

    if not os.path.isdir(grid_path):
        continue

    for file in tqdm(os.listdir(grid_path), desc=f"Processing {grid}"):

        if not file.endswith(".tif"):
            continue

        total += 1

        img_path = os.path.join(grid_path, file)
        mask_path = os.path.join(mask_root, file)

        # Check if mask exists
        if not os.path.exists(mask_path):
            missing_masks += 1
            continue

        # Copy image
        shutil.copy(img_path, os.path.join(out_img_dir, file))

        # Copy mask
        shutil.copy(mask_path, os.path.join(out_mask_dir, file))

print(f"\nTotal tiles: {total}")
print(f"Missing masks: {missing_masks}")
print(f"Final dataset size: {total - missing_masks}")