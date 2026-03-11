import os
import rasterio
import numpy as np
from scipy.ndimage import binary_opening, binary_closing
from skimage.measure import label

input_root = "maharashtra_sentinel_outputs"
mask_root = "maharashtra_water_masks"

os.makedirs(mask_root, exist_ok=True)

print("\n==============================")
print(" NDWI Water Mask Generation ")
print("==============================\n")

for grid in os.listdir(input_root):

    grid_path = os.path.join(input_root, grid)

    if not os.path.isdir(grid_path):
        continue

    for file in os.listdir(grid_path):

        if not file.endswith("_S2_NDWI.tif"):
            continue

        ndwi_path = os.path.join(grid_path, file)

        print("Processing:", ndwi_path)

        with rasterio.open(ndwi_path) as src:

            ndwi = src.read(1).astype(float)

            # dataMask band
            data_mask = src.read(2)

            profile = src.profile

            # Remove invalid pixels
            ndwi[data_mask == 0] = np.nan

            # Water detection (threshold for coastal regions)
            water = (ndwi > 0) & np.isfinite(ndwi)

            water_mask = water.astype(np.uint8)

        # ---------------------------------
        # CLEANING STEP 1: Remove noise
        # ---------------------------------

        water_mask = binary_opening(
            water_mask,
            structure=np.ones((3, 3))
        )

        # ---------------------------------
        # CLEANING STEP 2: Fill holes
        # ---------------------------------

        water_mask = binary_closing(
            water_mask,
            structure=np.ones((5, 5))
        )

        # ---------------------------------
        # CLEANING STEP 3: Remove tiny regions
        # ---------------------------------

        labeled = label(water_mask)

        for region in np.unique(labeled):

            region_size = np.sum(labeled == region)

            if region_size < 100:
                water_mask[labeled == region] = 0

        # Convert mask to 0 / 255 format
        water_mask = (water_mask * 255).astype(np.uint8)

        # ---------------------------------
        # SAVE CLEANED MASK
        # ---------------------------------

        output_grid = os.path.join(mask_root, grid)
        os.makedirs(output_grid, exist_ok=True)

        output_file = os.path.join(
            output_grid,
            file.replace("_S2_NDWI.tif", "_water_mask.tif")
        )

        profile.update(
            dtype=rasterio.uint8,
            count=1,
            compress="lzw"
        )

        with rasterio.open(output_file, "w", **profile) as dst:
            dst.write(water_mask, 1)

        print("Clean mask saved:", output_file)

print("\nAll water masks generated and cleaned.\n")