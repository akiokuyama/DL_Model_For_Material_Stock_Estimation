library(raster)
library(sf)
library(doSNOW)
ras = "C:/Users/SUS/My Drive/MS Estimation/StPaul/Original_Data/Satellite_Image2021/StPaul.tif"
ras = brick(ras)
#grid_polygons = "C:/Users/SUS/My Drive/MS Estimation/StPaul/2021/Created_Data/Material_Use_Parcel/MatarialUseParcel.shp"
#out_folder = "C:/Users/SUS/My Drive/MS Estimation/StPaul/2021/Created_Data/Satellite_Image/Parcel_Size/Parcel_Size_16Bit"
shp = read_sf("C:/Users/SUS/My Drive/MS Estimation/StPaul/2021/Created_Data/Grid/ParcelSize/Parcel_Grid_2021.shp")

#shp = st_transform(shp,ras)

cl = makeCluster(12)
registerDoSNOW(cl)

out_folder = "C:/Users/SUS/My Drive/MS Estimation/StPaul/2021/Created_Data/Satellite_Image/Parcel_Size/Parcel_Size_16Bit"

# Function to create a meaningful filename for each masked raster
create_filename <- function(folder, fid) {
  return(paste0(folder, "/StPaul2021_", fid, ".tif"))
}

# Split raster based on polygons
out = foreach(i = 1:nrow(shp)) %dopar% {
  library(raster)
  library(sf)
  
  subshp = shp[i,]
  fid_value = subshp$FID[[1]]  # Extract FID value
  
  # Crop raster based on polygon extent and mask it with the polygon shape
  sub = mask(crop(ras, subshp), subshp)
  
  # Write the masked raster to a file
  outfile = create_filename(out_folder, fid_value)
  writeRaster(sub, filename = outfile, format="GTiff", dataType="INT1U", overwrite=TRUE)
}

stopCluster(cl)


##### Change the splited image into 8bit
# Use the python code to change the pixel depth from 32bits to 8bits


