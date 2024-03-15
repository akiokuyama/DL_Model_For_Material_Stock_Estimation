import os
import arcpy
from multiprocessing import Pool, freeze_support

def convert_raster_to_8bit(args):
    raster, final_output_folder = args  # Unpack the tuple into individual variables
    
    # Extract the ID from the original raster name
    raster_id = raster.split('_')[-1].replace('.TIF', '')  # Assuming the ID is always after the last underscore
    
    # Create the new name with the "_3Bands" suffix
    new_name = f"StPaul2021_{raster_id}_8Bit.TIF"
    output_raster_path = os.path.join(final_output_folder, new_name)
    
    # Check if the raster has already been processed. If yes, skip.
    if os.path.exists(output_raster_path):
        print(f"Raster {raster} already processed. Skipping...")
        return

    print(f"Processing raster: {raster}")
    
    # Use Copy Raster to change pixel depth
    arcpy.CopyRaster_management(raster, output_raster_path, "", "", "", "", "", "8_BIT_UNSIGNED")

def main():
    out_folder = "C:/Users/SUS/My Drive/MS Estimation/StPaul/2021/Created_Data/Satellite_Image/Parcel_Size/Parcel_Size_16Bit"
    
    # List all the split rasters
    arcpy.env.workspace = out_folder
    split_rasters = [os.path.join(out_folder, raster) for raster in arcpy.ListRasters("StPaul2021_*")]

    final_output_folder = "C:/Users/SUS/My Drive/MS Estimation/StPaul/2021/Created_Data/Satellite_Image/Parcel_Size/Parcel_Size_8Bit"
    
    # Prepare arguments as tuples
    args = [(raster, final_output_folder) for raster in split_rasters]
    
    num_cores = 14  # Change this to adjust the number of cores used
    with Pool(num_cores) as pool:
        results = pool.map(convert_raster_to_8bit, args)

    print("All rasters processed.")

if __name__ == '__main__':
    freeze_support()  # Needed for Windows
    main()
