import rasterio
from rasterio.merge import merge
import glob
import pandas as pd
import os
import traceback

# Load metadata
df = pd.read_csv("tile_metadata.csv")

# Output folder
output_folder = "merged_grids"
os.makedirs(output_folder, exist_ok=True)

success = 0
failed = 0


def merge_grid(grid_id):

    global success, failed

    try:

        grid_id = int(grid_id)

        pattern = f"dynamic_world_tiles_local/grid_{grid_id}_tile_*.tif"
        files = glob.glob(pattern)

        if len(files) == 0:
            print(f"SKIPPED: No tiles found for Grid {grid_id}")
            failed += 1
            return

        print(f"Merging Grid {grid_id} ({len(files)} tiles)...")

        src_files = []

        for f in files:
            src = rasterio.open(f)
            src_files.append(src)

        mosaic, transform = merge(src_files)

        out_meta = src_files[0].meta.copy()

        out_meta.update({
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": transform,
            "driver": "GTiff"
        })

        output_path = f"{output_folder}/merged_grid_{grid_id}.tif"

        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(mosaic)

        # Close files
        for src in src_files:
            src.close()

        print(f"SUCCESS: Merged → {output_path}")
        success += 1

    except Exception as e:

        print(f"FAILED: Grid {grid_id}")
        print(traceback.format_exc())
        failed += 1


# Remove duplicate grid IDs
grid_ids = df["grid_id"].unique()

for grid_id in grid_ids:
    merge_grid(grid_id)


print("\n==============================")
print(f"Total Success: {success}")
print(f"Total Failed: {failed}")
print("==============================")
