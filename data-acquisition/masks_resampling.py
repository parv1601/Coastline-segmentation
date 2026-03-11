import os
import rasterio
from rasterio.warp import reproject
from rasterio.enums import Resampling
import numpy as np

sar_root = "kerala_sentinel_outputs"
mask_root = "kerala_water_masks"
output_root = "kerala_water_masks_resampled"

os.makedirs(output_root, exist_ok=True)

print("\n===================================")
print(" Resampling Water Masks to SAR Grid")
print("===================================\n")

for grid in sorted(os.listdir(sar_root)):

    sar_grid = os.path.join(sar_root, grid)
    mask_grid = os.path.join(mask_root, grid)

    if not os.path.isdir(sar_grid) or not os.path.isdir(mask_grid):
        continue

    sar_file = None
    mask_file = None

    for f in os.listdir(sar_grid):
        if f.endswith("_S1_SAR_GRD.tif"):
            sar_file = os.path.join(sar_grid, f)

    for f in os.listdir(mask_grid):
        if f.endswith("_water_mask.tif"):
            mask_file = os.path.join(mask_grid, f)

    if sar_file is None or mask_file is None:
        print(grid, "→ missing SAR or mask")
        continue

    with rasterio.open(sar_file) as sar, rasterio.open(mask_file) as mask:

        sar_profile = sar.profile
        sar_shape = (sar.height, sar.width)

        resampled_mask = np.zeros(sar_shape, dtype=np.uint8)

        reproject(
            source=rasterio.band(mask, 1),
            destination=resampled_mask,
            src_transform=mask.transform,
            src_crs=mask.crs,
            dst_transform=sar.transform,
            dst_crs=sar.crs,
            resampling=Resampling.nearest
        )

    output_grid = os.path.join(output_root, grid)
    os.makedirs(output_grid, exist_ok=True)

    output_file = os.path.join(
        output_grid,
        os.path.basename(mask_file).replace("_water_mask", "_water_mask_resampled")
    )

    sar_profile.update(
        dtype=rasterio.uint8,
        count=1,
        compress="lzw"
    )

    with rasterio.open(output_file, "w", **sar_profile) as dst:
        dst.write(resampled_mask, 1)

    print("Resampled mask created:", output_file)

print("\nResampling complete.\n")