import os
import rasterio

sar_root = "goa_sentinel_outputs"
mask_root = "goa_water_masks_resampled"

print("\n===================================")
print(" SAR vs Water Mask Alignment Check")
print("===================================\n")

for grid in sorted(os.listdir(sar_root)):

    sar_grid_path = os.path.join(sar_root, grid)

    if not os.path.isdir(sar_grid_path):
        continue

    # find SAR file
    sar_file = None
    for f in os.listdir(sar_grid_path):
        if f.endswith("_S1_SAR_GRD.tif"):
            sar_file = os.path.join(sar_grid_path, f)

    # find mask file
    mask_grid_path = os.path.join(mask_root, grid)

    if not os.path.exists(mask_grid_path):
        print(grid, "→ mask folder missing")
        continue

    mask_file = None
    for f in os.listdir(mask_grid_path):
        if f.endswith("_water_mask_resampled.tif"):
            mask_file = os.path.join(mask_grid_path, f)

    if sar_file is None or mask_file is None:
        print(grid, "→ missing SAR or mask")
        continue

    with rasterio.open(sar_file) as sar, rasterio.open(mask_file) as mask:

        print("-----------------------------------")
        print("Grid:", grid)

        print("SAR shape :", sar.height, sar.width)
        print("Mask shape:", mask.height, mask.width)

        print("SAR CRS :", sar.crs)
        print("Mask CRS:", mask.crs)

        print("SAR resolution :", sar.res)
        print("Mask resolution:", mask.res)

        print("SAR bounds :", sar.bounds)
        print("Mask bounds:", mask.bounds)

        if sar.crs == mask.crs:
            print("CRS match: YES")
        else:
            print("CRS match: NO")

        if sar.transform == mask.transform:
            print("Transform match: YES")
        else:
            print("Transform match: NO")

print("\nAlignment check complete.\n")