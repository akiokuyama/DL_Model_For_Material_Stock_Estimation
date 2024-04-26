import rasterio
from rasterio.enums import Resampling

def remove_nir_band(input_file, output_file):
    # Open the input TIFF file
    with rasterio.open(input_file) as src:
        # Check if there are at least 4 bands (Assuming bands are ordered as Red, Green, Blue, NIR)
        if src.count < 4:
            raise ValueError("The input file does not have enough bands.")

        # Read the first three bands (Red, Green, Blue)
        red, green, blue = src.read([1, 2, 3])

        # Define metadata for the new file without the NIR band
        out_meta = src.meta.copy()
        out_meta.update({
            'count': 3,  # Number of bands
            'dtype': 'uint8'  # Adjust data type if needed
        })

        # Write the new file without the NIR band
        with rasterio.open(output_file, 'w', **out_meta) as dst:
            dst.write(red, 1)
            dst.write(green, 2)
            dst.write(blue, 3)
            print(f"File saved without the NIR band: {output_file}")

# Example usage
input_tif = ""
output_tif = ""
remove_nir_band(input_tif, output_tif)
