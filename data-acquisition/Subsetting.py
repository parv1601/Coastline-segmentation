import os
import esa_snappy
from esa_snappy import ProductIO, GPF

# Enable SNAP operator registry
GPF.getDefaultInstance().getOperatorSpiRegistry().loadOperatorSpis()

# Java HashMap
HashMap = esa_snappy.jpy.get_type('java.util.HashMap')
Integer = esa_snappy.jpy.get_type('java.lang.Integer')

# Input directory
input_root = "kerala_sentinel_outputs"

# Output directory
output_root = "kerala_sentinel_outputs_filtered"

os.makedirs(output_root, exist_ok=True)

print("\n==============================")
print(" Sentinel-1 Speckle Filtering ")
print("==============================\n")

for grid_folder in sorted(os.listdir(input_root)):

    input_grid_path = os.path.join(input_root, grid_folder)

    if not os.path.isdir(input_grid_path):
        continue

    output_grid_path = os.path.join(output_root, grid_folder)
    os.makedirs(output_grid_path, exist_ok=True)

    print(f"\nProcessing Grid: {grid_folder}")

    for file in os.listdir(input_grid_path):

        if not file.endswith("_S1_SAR_GRD.tif"):
            continue

        input_path = os.path.join(input_grid_path, file)

        output_file = os.path.join(
            output_grid_path,
            file.replace(".tif", "_filtered")
        )

        # Skip if already processed
        if os.path.exists(output_file + ".tif"):
            print("Skipping (already processed):", file)
            continue

        print("\n----------------------------------")
        print("Input:", input_path)

        try:

            # Load product
            product = ProductIO.readProduct(input_path)

            if product is None:
                print("Could not read product")
                continue

            # Print bands (useful for debugging)
            band_names = list(product.getBandNames())
            print("Bands:", band_names)

            # Speckle filter parameters
            parameters = HashMap()
            parameters.put("filter", "Refined Lee")
            parameters.put("filterSizeX", Integer(5))
            parameters.put("filterSizeY", Integer(5))
            parameters.put("estimateENL", True)
            

            # Apply filter
            filtered_product = GPF.createProduct(
                "Speckle-Filter",
                parameters,
                product
            )

            print("Saving to:", output_file)

            ProductIO.writeProduct(
                filtered_product,
                output_file,
                "GeoTIFF"
            )

            # Free memory (important for large datasets)
            product.dispose()
            filtered_product.dispose()

            print("Finished:", file)

        except Exception as e:
            print("Error processing:", file)
            print(e)

print("\nAll filtering completed successfully.\n")