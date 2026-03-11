import ee
import pandas as pd
import os
import requests
import traceback
import math

ee.Initialize(project='coastline-project-487413')

# INPUT GRID FILE
csv_file = "kerala_grid_bbox.csv"

# OUTPUT
output_folder = "dynamic_world_tiles_KL_2yr"
os.makedirs(output_folder, exist_ok=True)

df = pd.read_csv(csv_file)

# Dynamic World dataset
dw = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1") \
        .filterDate("2024-01-01", "2025-12-31") \
        .select("label") \
        .mode()

# SAFE TILE SIZE
tile_size_deg = 2.5

success = 0
failed = 0

tile_metadata = []


# SPLIT FUNCTION
def split_grid(min_lon, min_lat, max_lon, max_lat):

    lon_steps = math.ceil((max_lon - min_lon) / tile_size_deg)
    lat_steps = math.ceil((max_lat - min_lat) / tile_size_deg)

    tiles = []

    for i in range(lon_steps):
        for j in range(lat_steps):

            tile_min_lon = min_lon + i * tile_size_deg
            tile_max_lon = min(tile_min_lon + tile_size_deg, max_lon)

            tile_min_lat = min_lat + j * tile_size_deg
            tile_max_lat = min(tile_min_lat + tile_size_deg, max_lat)

            tiles.append([
                tile_min_lon,
                tile_min_lat,
                tile_max_lon,
                tile_max_lat
            ])

    return tiles


# MAIN LOOP
for idx, row in df.iterrows():

    grid_id = int(row["grid_id"])

    print(f"\nProcessing Grid {grid_id}")

    tiles = split_grid(
        row["min_lon"],
        row["min_lat"],
        row["max_lon"],
        row["max_lat"]
    )

    print(f"Split into {len(tiles)} tiles")

    for tile_id, bbox in enumerate(tiles):

        print(f"Downloading Grid {grid_id}, Tile {tile_id}")

        region = ee.Geometry.Rectangle(bbox)

        filename = f"{output_folder}/grid_{grid_id}_tile_{tile_id}.tif"

        try:
            # "#419BDF",  # water
            # "#397D49",  # trees
            # "#88B053",  # grass
            # "#7A87C6",  # flooded veg
            # "#E49635",  # crops
            # "#DFC35A",  # shrub
            # "#C4281B",  # built
            # "#A59B8F",  # bare
            # "#B39FE1"   # snow
            vis = {
                'min': 0,
                'max': 8,
                'palette': [
                    "#419BDF", "#397D49", "#88B053", "#7A87C6",
                    "#E49635", "#DFC35A", "#C4281B", "#A59B8F", "#B39FE1"
                ]
            }

            image = dw.visualize(**vis).clip(region)

            url = image.getDownloadURL({
                'region': region,
                'scale': 10,
                'format': 'GEO_TIFF'
            })

            response = requests.get(url, stream=True)

            if response.status_code == 200:

                with open(filename, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)

                print(f"SUCCESS")

                tile_metadata.append({
                    "grid_id": grid_id,
                    "tile_id": tile_id,
                    "min_lon": bbox[0],
                    "min_lat": bbox[1],
                    "max_lon": bbox[2],
                    "max_lat": bbox[3]
                })

                success += 1

            else:

                print(f"FAILED HTTP {response.status_code}")
                failed += 1

        except Exception:

            print(f"FAILED Exception")
            print(traceback.format_exc())

            failed += 1


# SAVE TILE METADATA
pd.DataFrame(tile_metadata).to_csv("tile_metadata_kerala_2yr.csv", index=False)


print("\n======================")
print("DOWNLOAD COMPLETE")
print("Success:", success)
print("Failed:", failed)
print("======================")
