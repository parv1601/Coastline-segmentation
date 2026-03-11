import os
import rasterio
import numpy as np

base_dir = "goa_sentinel_outputs"

print("\n===================================")
print(" Goa Grid TIFF Inspection + NDWI Analysis")
print("===================================\n")

for grid_folder in sorted(os.listdir(base_dir)):

    grid_path = os.path.join(base_dir, grid_folder)

    if not os.path.isdir(grid_path):
        continue

    print("===================================")
    print("Grid:", grid_folder)
    print("===================================")

    for file in os.listdir(grid_path):

        if not file.endswith(".tif"):
            continue

        file_path = os.path.join(grid_path, file)

        try:
            with rasterio.open(file_path) as src:

                # -------- TIFF BAND INFO --------
                print("\nFile:", file)
                print("Bands:", src.count)
                print("Size:", src.width, "x", src.height)
                print("Data type:", src.dtypes)

                # -------- NDWI ANALYSIS --------
                if file.endswith("_S2_NDWI.tif"):

                    ndwi = src.read(1).astype(float)   # NDWI
                    mask = src.read(2)                 # dataMask

                    ndwi[mask == 0] = np.nan

                    ndwi = ndwi[np.isfinite(ndwi)]

                    if ndwi.size > 0:
                        print("\nNDWI Statistics")
                        print("Min:", np.min(ndwi))
                        print("Max:", np.max(ndwi))
                        print("Mean:", np.mean(ndwi))
                        print("Std:", np.std(ndwi))
                        print("Total Pixels:", ndwi.size)
                    else:
                        print("NDWI contains no valid pixels")

        except Exception as e:
            print("Error reading:", file)
            print(e)

    print("\n-----------------------------------\n")

print("Inspection + NDWI Analysis Complete.\n")