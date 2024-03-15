# Delete ".tif" in the middle of "StPaul2021_{id}.tif_8Bit.TIF"
import os

directory = "C:/Users/SUS/My Drive/MS Estimation/StPaul/2021/Created_Data/Satellite_Image/Parcel_Size/Parcel_Size_8Bit/3Bands"  

# List all files in the directory
file_list = os.listdir(directory)

for filename in file_list:
    if filename.endswith(".tif_8Bit_3bands.tif"):
        new_filename = filename.replace(".tif_8Bit_3bands.tif", "_8Bit_3bands.tif")
        old_filepath = os.path.join(directory, filename)
        new_filepath = os.path.join(directory, new_filename)
        os.rename(old_filepath, new_filepath)
        print(f"Renamed: {filename} to {new_filename}")