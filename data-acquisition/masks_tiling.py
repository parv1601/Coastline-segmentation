import os
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

# Paths
sar_root = "kerala_tiles_output_sentinel"
mask_root = "kerala_water_masks_resampled"
output_mask_dir = "kerala_processed_masks"

os.makedirs(output_mask_dir, exist_ok=True)

for grid in os.listdir(sar_root):
    sar_grid_path = os.path.join(sar_root, grid)
    mask_grid_path = os.path.join(mask_root, grid)

    # Find mask file (only 1 per grid)
    mask_file = [f for f in os.listdir(mask_grid_path) if f.endswith(".tif")][0]
    mask_path = os.path.join(mask_grid_path, mask_file)

    with rasterio.open(mask_path) as mask_src:

        for tile_name in tqdm(os.listdir(sar_grid_path), desc=f"{grid}"):

            if not tile_name.endswith(".tif"):
                continue

            # Example: grid1_0_182.tif
            parts = tile_name.replace(".tif", "").split("_")
            row = int(parts[1])
            col = int(parts[2])

            # Extract corresponding mask tile
            window = Window(col, row, 512, 512)
            mask_tile = mask_src.read(1, window=window)

            # Convert 255 → 1
            mask_tile = (mask_tile > 0).astype("uint8")

            # Save
            out_path = os.path.join(output_mask_dir, tile_name)

            with rasterio.open(
                out_path,
                'w',
                driver='GTiff',
                height=512,
                width=512,
                count=1,
                dtype='uint8'
            ) as dst:
                dst.write(mask_tile, 1)