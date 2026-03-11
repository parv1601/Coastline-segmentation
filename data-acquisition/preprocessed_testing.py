import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt

base_dir = "goa_sentinel_outputs"

print("\n Sentinel-1 SAR Preprocessed Data Verification\n")

for grid_folder in sorted(os.listdir(base_dir)):

    grid_path = os.path.join(base_dir, grid_folder)

    if not os.path.isdir(grid_path):
        continue

    sar_files = [f for f in os.listdir(grid_path) if f.endswith("_S1_SAR_GRD.tif")]
    if not sar_files:
        continue

    sar_path = os.path.join(grid_path, sar_files[0])

    print("===================================================")
    print(f" {grid_folder}")
    print(f"File: {sar_files[0]}")

    with rasterio.open(sar_path) as src:

        vv = src.read(1).astype(float)
        vh = src.read(2).astype(float)

        vv = vv[np.isfinite(vv)]
        vh = vh[np.isfinite(vh)]

        # Detect scale
        scale_type = "LINEAR" if vv.max() > 1 else "DB"
        print("\nDetected SAR scale:", scale_type)

        # Convert to dB
        vv_db = 10 * np.log10(vv + 1e-10)
        vh_db = 10 * np.log10(vh + 1e-10)

        print("\n VV Statistics (Linear)")
        print("Min:", vv.min())
        print("Max:", vv.max())
        print("Mean:", vv.mean())
        print("Std Dev:", vv.std())

        print("\n VV Statistics (dB)")
        print("Min:", vv_db.min())
        print("Max:", vv_db.max())
        print("Mean:", vv_db.mean())

        print("\n VH Mean (dB):", vh_db.mean())

        # Calibration check
        if -40 < vv_db.min() and vv_db.max() < 30:
            print(" Backscatter range confirms radiometric calibration")

        if vv_db.mean() > vh_db.mean():
            print(" VV stronger than VH → correct SAR physics")

        # Water detection
        water_mask = vv < 0.02
        if water_mask.sum() > 0:
            print(" Water detected using threshold 0.02 (linear)")

        # HISTOGRAM
        plt.figure(figsize=(6,4))
        plt.hist(vv_db.flatten(), bins=150)
        plt.title(f"{grid_folder} VV Histogram (dB)")
        plt.xlabel("Backscatter (dB)")
        plt.ylabel("Pixel Count")

        plt.show(block=False)
        plt.pause(2)
        plt.close()

        # WATER MASK
        plt.figure(figsize=(5,5))
        plt.imshow(water_mask.reshape(src.height, src.width), cmap="gray")
        plt.title(f"{grid_folder} Water Mask")
        plt.axis("off")

        plt.show(block=False)
        plt.pause(2)
        plt.close()

print("\n Analysis Complete")