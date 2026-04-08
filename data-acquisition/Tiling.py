import esa_snappy
from esa_snappy import jpy, ProductIO, GPF
import os

# INPUT: One state directory ONLY
root_dir = "maharashtra_sentinel_outputs_filtered"

# OUTPUT DIRECTORY
output_root = "maharashtra_tiles_output_sentinel"
os.makedirs(output_root, exist_ok=True)

# SNAP setup
HashMap = jpy.get_type('java.util.HashMap')
GPF.getDefaultInstance().getOperatorSpiRegistry().loadOperatorSpis()

tile_size = 512

print(f"Processing STATE folder: {root_dir}")

# =========================
# LOOP OVER GRIDS
# =========================
grid_dirs = [d for d in os.listdir(root_dir) if d.startswith("grid_")]

for grid in grid_dirs:

    grid_path = os.path.join(root_dir, grid)

    # Convert grid_1 → grid1
    grid_name_fixed = grid.replace("_", "")

    input_file = os.path.join(
        grid_path,
        f"{grid_name_fixed}_S1_SAR_GRD_filtered.tif"
    )

    if not os.path.exists(input_file):
        print(f"Missing: {input_file}")
        continue

    print(f"\nReading: {input_file}")

    product = ProductIO.readProduct(input_file)

    img_width = product.getSceneRasterWidth()
    img_height = product.getSceneRasterHeight()

    print(f"Size: {img_width} x {img_height}")

    # =========================
    # DEFINE 4 TILE POSITIONS
    # =========================
    positions = [
        (0, 0),
        (img_width - tile_size, 0),
        (0, img_height - tile_size),
        (img_width - tile_size, img_height - tile_size)
    ]

    # Create output folder for this grid
    grid_output_dir = os.path.join(output_root, grid)
    os.makedirs(grid_output_dir, exist_ok=True)

    # =========================
    # EXTRACT TILES
    # =========================
    for (x, y) in positions:

        # Safety check
        if x < 0 or y < 0:
            continue

        print(f"Tile: {x}, {y}")

        parameters = HashMap()
        parameters.put('copyMetadata', True)
        parameters.put('region', f"{x},{y},{tile_size},{tile_size}")

        product_subset = GPF.createProduct('Subset', parameters, product)

        output_name = f"{grid_name_fixed}_{x}_{y}"
        output_path = os.path.join(grid_output_dir, output_name)

        ProductIO.writeProduct(product_subset, output_path, 'GeoTIFF')

print("\n====================")
print("TILING COMPLETE")
print("====================")