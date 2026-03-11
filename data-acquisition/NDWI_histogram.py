import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt

base_dir = "goa_sentinel_outputs"
hist_output_dir = "goa_ndwi_histograms"

os.makedirs(hist_output_dir, exist_ok=True)

print("\n===================================")
print(" NDWI Histogram Analysis")
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

                ndwi = src.read(1).astype(float)
                mask = src.read(2)

                # remove invalid pixels
                ndwi[mask == 0] = np.nan
                valid_ndwi = ndwi[np.isfinite(ndwi)]

                if valid_ndwi.size == 0:
                    print("Grid:", grid_folder, "→ No valid pixels")
                    continue

                # Histogram
                hist, bin_edges = np.histogram(valid_ndwi, bins=200, range=(-1, 1))

                # Peak detection
                peak_index = np.argmax(hist)
                peak_value = (bin_edges[peak_index] + bin_edges[peak_index+1]) / 2

                print("-----------------------------------")
                print("Grid:", grid_folder)
                print("Peak NDWI value:", round(peak_value, 4))

                # Plot histogram
                plt.figure(figsize=(8,5))
                plt.hist(valid_ndwi, bins=200, range=(-1,1))
                plt.axvline(peak_value, linestyle="--")
                plt.title(f"NDWI Histogram - {grid_folder}")
                plt.xlabel("NDWI Value")
                plt.ylabel("Pixel Count")

                save_path = os.path.join(hist_output_dir, f"{grid_folder}_hist.png")
                plt.savefig(save_path)
                plt.close()

        except Exception as e:
            print("Error reading:", ndwi_path)
            print(e)

print("\nHistogram analysis complete.\n")