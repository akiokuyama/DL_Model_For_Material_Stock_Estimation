rm(list=ls())


# Import Librarlies -------------------------------------------------------
library(sf)
library(raster)
library(reshape2)
library(stringr)
library(fasterize)
library(ggplot2)
library(tidyverse)
library(tibble)
library(rgdal)
library(dplyr)

split_footprint_path="C:/Users/SUS/My Drive/MS Estimation/StPaul/2021/Created_Data/Material_Use_Parcel/MatarialUseParcel.shp"
cityboundary_path="C:/Users/SUS/My Drive/MS Estimation/StPaul/Original_Data/City_Boundary/StPaul/StPaul_Bounary.shp"
output_path="C:/Users/SUS/My Drive/MS Estimation/StPaul/2021/Created_Data/Grid/ParcelSize"



# Create grid (same as parcel) and calculate MI for each pixel -----------------------------
split_footprint = read_sf(split_footprint_path)
cityboundary = read_sf(cityboundary_path)

cityboundary <- st_transform(cityboundary, st_crs(split_footprint))

parcel_grid <- split_footprint[, c("FID", "geometry")]


st_write(parcel_grid, paste0(output_path, "/Parcel_Grid_2021.shp"))


