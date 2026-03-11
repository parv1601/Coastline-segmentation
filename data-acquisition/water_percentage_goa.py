import os
import rasterio
import numpy as np

base_dir = "goa_sentinel_outputs"

print("\n===================================")
print(" NDWI Water Percentage per Grid")
print("===================================\n")

for grid_folder in sorted(os.listdir(base_dir)):

    grid_path = os.path.join(base_dir, grid_folder)

    if not os.path.isdir(grid_path):
        continue

    for file in os.listdir(grid_path):

        if not file.endswith("_S2_NDWI.tif"):
            continue

        ndwi_path = os.path.join(grid_path, file)

        try:
            with rasterio.open(ndwi_path) as src:

                # Read NDWI and mask
                ndwi = src.read(1).astype(float)
                mask = src.read(2)

                # Remove invalid pixels
                ndwi[mask == 0] = np.nan

                # Keep valid pixels only
                valid_pixels = ndwi[np.isfinite(ndwi)]

                total_pixels = valid_pixels.size

                if total_pixels == 0:
                    print("Grid:", grid_folder, "→ No valid pixels")
                    continue

                # Water pixels (NDWI > 0)
                water_pixels = np.sum(valid_pixels > 0.1)

                water_percent = (water_pixels / total_pixels) * 100

                print("-----------------------------------")
                print("Grid:", grid_folder)
                print("Total Valid Pixels:", total_pixels)
                print("Water Pixels:", water_pixels)
                print("Water Percentage: {:.2f}%".format(water_percent))

        except Exception as e:
            print("Error reading:", ndwi_path)
            print(e)

print("\nWater analysis complete.\n")