import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt

mask_dir = "kerala_processed_masks"   # <-- CHANGE THIS

print("\n===================================")
print(" KERALA TILE WATER DISTRIBUTION ")
print("===================================\n")

water_percentages = []

zero_water_tiles = 0

for file in os.listdir(mask_dir):

    if not file.endswith(".tif"):
        continue

    mask_path = os.path.join(mask_dir, file)

    with rasterio.open(mask_path) as src:
        mask = src.read(1)

        # ensure binary
        mask = (mask > 0).astype(np.uint8)

        total_pixels = mask.size
        water_pixels = np.sum(mask)

        water_percent = (water_pixels / total_pixels) * 100
        water_percentages.append(water_percent)

        if water_pixels == 0:
            zero_water_tiles += 1

water_percentages = np.array(water_percentages)

# =========================
# PRINT STATS
# =========================
print("Total tiles:", len(water_percentages))
print("Mean water %:", np.mean(water_percentages))
print("Median water %:", np.median(water_percentages))

print("\n--- Distribution ---")
print("Tiles with 0% water:", zero_water_tiles)
print("Tiles with <1% water:", np.sum(water_percentages < 1))
print("Tiles with <5% water:", np.sum(water_percentages < 5))
print("Tiles with >20% water:", np.sum(water_percentages > 20))

# =========================
# HISTOGRAM
# =========================
plt.figure()
plt.hist(water_percentages, bins=30)
plt.title("Water % Distribution Across Tiles")
plt.xlabel("Water Percentage")
plt.ylabel("Number of Tiles")
plt.show()