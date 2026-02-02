import os
import pandas as pd
from sentinelhub import (
    SHConfig, BBox, CRS, bbox_to_dimensions,
    SentinelHubRequest, DataCollection, MimeType
)

# --------------------------
# 1) CONFIG (Copernicus Data Space OAuth)
# --------------------------
config = SHConfig()

# Ensure these are set in your environment or replace with strings for testing
config.sh_client_id = os.getenv("SH_CLIENT_ID")
config.sh_client_secret = os.getenv("SH_CLIENT_SECRET")

# CDSE Specific Endpoints
config.sh_base_url = "https://sh.dataspace.copernicus.eu"
config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

if not config.sh_client_id or not config.sh_client_secret:
    raise ValueError("Missing SH_CLIENT_ID / SH_CLIENT_SECRET environment variables")

# Force-defining collections to use CDSE service URL
# This prevents the library from defaulting to the old services.sentinel-hub.com
S2_L2A_CDSE = DataCollection.SENTINEL2_L2A.define_from(
    "s2-l2a-cdse", service_url=config.sh_base_url
)
S1_IW_CDSE = DataCollection.SENTINEL1_IW.define_from(
    "s1-iw-cdse", service_url=config.sh_base_url
)

# --------------------------
# 2) INPUT CSV
# --------------------------
csv_file = "goa_grid_bbox.csv"
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"Could not find {csv_file}")

df = pd.read_csv(csv_file)

# --------------------------
# 3) SETTINGS
# --------------------------
TIME_INTERVAL = ("2025-05-01", "2025-05-31")
RESOLUTION_S2 = 10
RESOLUTION_S1 = 20

OUT_FOLDER = "goa_sentinel_outputs"
os.makedirs(OUT_FOLDER, exist_ok=True)

# --------------------------
# 4) EVALSCRIPTS
# --------------------------
EVALSCRIPT_S2_NDVI = """
//VERSION=3
function setup() {
  return {
    input: ["B04", "B08", "dataMask"],
    output: { bands: 2, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(s) {
  let ndvi = (s.B08 - s.B04) / (s.B08 + s.B04 + 1e-6);
  return [ndvi, s.dataMask];
}
"""

EVALSCRIPT_S2_RGB = """
//VERSION=3
function setup() {
  return {
    input: ["B02", "B03", "B04", "dataMask"],
    output: { bands: 4, sampleType: "UINT16" }
  };
}
function evaluatePixel(s) {
  return [s.B04, s.B03, s.B02, s.dataMask];
}
"""

EVALSCRIPT_S1_VV_VH = """
//VERSION=3
function setup() {
  return {
    input: ["VV", "VH", "dataMask"],
    output: { bands: 3, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(s) {
  return [s.VV, s.VH, s.dataMask];
}
"""

# --------------------------
# HELPERS
# --------------------------
def make_bbox(row):
    return BBox(
        bbox=[row["min_lon"], row["min_lat"], row["max_lon"], row["max_lat"]],
        crs=CRS.WGS84
    )

def get_sizes(bbox):
    size_s2 = bbox_to_dimensions(bbox, resolution=RESOLUTION_S2)
    size_s1 = bbox_to_dimensions(bbox, resolution=RESOLUTION_S1)
    return size_s2, size_s1

def download_s2(grid_folder, bbox, size_s2):
    # NDVI Request
    req_ndvi = SentinelHubRequest(
        evalscript=EVALSCRIPT_S2_NDVI,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=S2_L2A_CDSE,
                time_interval=TIME_INTERVAL,
                mosaicking_order="mostRecent",
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size_s2,
        config=config,
        data_folder=grid_folder
    )
    req_ndvi.get_data(save_data=True)

    # RGB Request
    req_rgb = SentinelHubRequest(
        evalscript=EVALSCRIPT_S2_RGB,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=S2_L2A_CDSE,
                time_interval=TIME_INTERVAL,
                mosaicking_order="mostRecent",
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size_s2,
        config=config,
        data_folder=grid_folder
    )
    req_rgb.get_data(save_data=True)

def download_s1(grid_folder, bbox, size_s1):
    req_sar = SentinelHubRequest(
        evalscript=EVALSCRIPT_S1_VV_VH,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=S1_IW_CDSE,
                time_interval=TIME_INTERVAL,
                mosaicking_order="mostRecent",
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size_s1,
        config=config,
        data_folder=grid_folder
    )
    req_sar.get_data(save_data=True)

# --------------------------
# MAIN
# --------------------------
def main():
    for _, row in df.iterrows():
        grid_id = int(row["grid_id"])
        bbox = make_bbox(row)
        size_s2, size_s1 = get_sizes(bbox)

        grid_folder = os.path.join(OUT_FOLDER, f"grid_{grid_id}")
        os.makedirs(grid_folder, exist_ok=True)

        print(f"\n Grid {grid_id}: Processing Sentinel-2 and Sentinel-1 via CDSE...")
        try:
            download_s2(grid_folder, bbox, size_s2)
            download_s1(grid_folder, bbox, size_s1)
            print(f"Grid {grid_id} completed.")
        except Exception as e:
            print(f"Error downloading Grid {grid_id}: {e}")

    print("\n DONE: All Goa grids processed successfully.")

if __name__ == "__main__":
    main()