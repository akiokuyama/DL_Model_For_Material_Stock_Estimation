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
library(lidR)



## Read the parcel and define building use -------------------------------
## Read spatial data and Boundaries
Parcel_StPaul2021 = read_sf("C:/Users/SUS/My Drive/MS Estimation/StPaul/2021/Created_Data/Parcel_StPaul/Parcel_StPaul2021.shp")

Parcel_StPaul2021$FAm2 <- Parcel_StPaul2021$FIN_SQ_FT * 0.092903
## unique values in county code to check all the data are in Ramsey County
unique_values <- unique(Parcel_StPaul2021$CO_NAME)
print(unique_values)
# Create FID
Parcel_StPaul2021$FID <- 0:(nrow(Parcel_StPaul2021)-1)
## Delete uncessary Fields
## Print all field names in Parcel_StPaul
print(names(Parcel_StPaul2021))
Parcel_StPaul2021 <- Parcel_StPaul2021 %>%
  select(FID, ZIP, USECLASS1, USECLASS2, USECLASS3, USECLASS4, DWELL_TYPE, FIN_SQ_FT, YEAR_BUILT, NUM_UNITS, FAm2)

## Check unique values to categorise building due to their usage
# unique_USECLASS1 <- unique(Parcel_StPaul2021$USECLASS1)
# print(unique_USECLASS1)
# 
# unique_USECLASS2 <- unique(Parcel_StPaul2021$USECLASS2)
# print(unique_USECLASS2)
# 
# unique_USECLASS3 <- unique(Parcel_StPaul2021$USECLASS3)
# print(unique_USECLASS3)
# 
# unique_USECLASS4 <- unique(Parcel_StPaul2021$USECLASS4)
# print(unique_USECLASS4)
# 
# unique_DWELL_TYPE <- unique(Parcel_StPaul2021$DWELL_TYPE)
# print(unique_DWELL_TYPE)


## Residential bld
Parcel_StPaul2021$Bld_use <- ifelse(!is.na(Parcel_StPaul2021$DWELL_TYPE), 1, 0)
# what are in USECLASS1 when Bld_Use is 0
unique_values <- Parcel_StPaul2021 %>%
  filter(Bld_use == 0) %>%
  distinct(USECLASS1) %>%
  pull(USECLASS1)
print(unique_values)

Parcel_StPaul2021 <- Parcel_StPaul2021 %>%
  mutate(Bld_use = case_when(
    USECLASS1 %in% c("Res 1 unit", "Res 2-3 units", "Apt 4+ units", "Schools-Priv Res", "Church-Other Res",
                     "Church-Residence", "College-Priv Res", "Charit Inst-Res", "Res V Land") ~ 1,
    TRUE ~ 0
  ))

## Commercial bld
Non_Residential <- Parcel_StPaul2021[Parcel_StPaul2021$Bld_use != 1, ]
unique_USECLASS1_nonresi <- unique(Non_Residential$USECLASS1)
print(unique_USECLASS1_nonresi)

Parcel_StPaul2021 <- Parcel_StPaul2021 %>%
  mutate(Bld_use = ifelse(
    USECLASS1 %in% c("Commercial", "Muni Srvc Other", "Muni Srvc Ent", "Co Srvc Other",
                     "Co Srvc Ent", "Comm Services-Non Revenue", "Comm Serv-Donations Congr Charter"), 2, Bld_use))

## Industrial bld
Non_Resi_Com <- Parcel_StPaul2021[Parcel_StPaul2021$Bld_use != 1 & Parcel_StPaul2021$Bld_use != 2, ]
unique_USECLASS1_nonresi_com <- unique(Non_Resi_Com$USECLASS1)
print(unique_USECLASS1_nonresi_com)

Parcel_StPaul2021 <- Parcel_StPaul2021 %>%
  mutate(Bld_use = ifelse(
    USECLASS1 %in% c("Industrial"), 3, Bld_use))

## Educational bld
Non_Resi_Com_Ind <- Parcel_StPaul2021[Parcel_StPaul2021$Bld_use != 1 & Parcel_StPaul2021$Bld_use != 2 & Parcel_StPaul2021$Bld_use != 3, ]
unique_USECLASS1_nonresi_com_ind <- unique(Non_Resi_Com_Ind$USECLASS1)
print(unique_USECLASS1_nonresi_com_ind)

Parcel_StPaul2021 <- Parcel_StPaul2021 %>%
  mutate(Bld_use = ifelse(
    USECLASS1 %in% c("Colleges-Private", "Schools-Public", "Colleges-Public", "Schools-Private",
                     "Apprenticeship Training Facilities"), 4, Bld_use))

## Other use of bld
Non_Resi_Com_Ind_Edu <- Parcel_StPaul2021[Parcel_StPaul2021$Bld_use != 1 & Parcel_StPaul2021$Bld_use != 2 & Parcel_StPaul2021$Bld_use != 3 & Parcel_StPaul2021$Bld_use != 4, ]
unique_USECLASS1_nonresi_com_ind_edu <- unique(Non_Resi_Com_Ind_Edu$USECLASS1)
print(unique_USECLASS1_nonresi_com_ind_edu)
unique_Bld_use_nonresi_com_ind_edu <- unique(Non_Resi_Com_Ind_Edu$Bld_use) # check if others have 0 for bld_use
unique_Bld_use_nonresi_com_ind_edu # check if others have 0 for bld_use


desired_order <- c("FID", "FAm2", "Bld_use", "FIN_SQ_FT","YEAR_BUILT")
## Reorder the columns based on the desired order
Parcel_StPaul2021 <- Parcel_StPaul2021 %>% 
  select(all_of(desired_order), everything())


## check the categories of building use
# filtered_Parcel_StPaul <- Parcel_StPaul %>%
#   filter(Bld_use  %in% 4)
# unique_values_filtered <- unique(filtered_Parcel_StPaul$PR_TYP_NM1)
# print(unique_values_filtered)

## save the parcel data with building use
st_write(Parcel_StPaul2021, "C:/Users/SUS/My Drive/MS Estimation/StPaul/2021/Created_Data/Parcel_StPaul/Parcel_StPaul2021_withBldUse.shp")


## Count the number of bld-use
filtered_sf_data_1 <- Parcel_StPaul2021[Parcel_StPaul2021$Bld_use == 1, ]
filtered_sf_data_2 <- Parcel_StPaul2021[Parcel_StPaul2021$Bld_use == 2, ]
filtered_sf_data_3 <- Parcel_StPaul2021[Parcel_StPaul2021$Bld_use == 3, ]
filtered_sf_data_4 <- Parcel_StPaul2021[Parcel_StPaul2021$Bld_use == 4, ]
filtered_sf_data_0 <- Parcel_StPaul2021[Parcel_StPaul2021$Bld_use == 0, ]
count_polygons_with_value_1 <- nrow(filtered_sf_data_1)
count_polygons_with_value_2 <- nrow(filtered_sf_data_2)
count_polygons_with_value_3 <- nrow(filtered_sf_data_3)
count_polygons_with_value_4 <- nrow(filtered_sf_data_4)
count_polygons_with_value_0 <- nrow(filtered_sf_data_0)
## Print the counts
cat("Number of polygons with 'Bld_use' equal to 1:", count_polygons_with_value_1, "/n")
cat("Number of polygons with 'Bld_use' equal to 2:", count_polygons_with_value_2, "/n")
cat("Number of polygons with 'Bld_use' equal to 3:", count_polygons_with_value_3, "/n")
cat("Number of polygons with 'Bld_use' equal to 4:", count_polygons_with_value_4, "/n")
cat("Number of polygons with 'Bld_use' equal to 0:", count_polygons_with_value_0, "/n")

# Estimate Material Use -------------------------------------------------------
## Import MI CSv file
MI = read.csv("C:/Users/SUS/My Drive/MS Estimation/StPaul/Original_Data/Material_Intensity/Material_Intensity.csv")
StPaul_footprint = read_sf("C:/Users/SUS/My Drive/MS Estimation/StPaul/2021/Created_Data/Parcel_StPaul/Parcel_StPaul2021_withBldUse.shp")

MI$Concrete <- MI$Concrete / 1000  # Change the unit to ton from kg
MI$Cement <- MI$Cement / 1000  # Change the unit to ton from kg


for (i in 1:nrow(MI)) { # Create variables
  type_name <- MI[i, "Material.Intenstiy..kg.m2."]
  assign(paste(type_name, "Concrete", sep = "_"), MI[i, "Concrete"])
  assign(paste(type_name, "Cement", sep = "_"), MI[i, "Cement"])
}

# 1: Residential, 2: Commercial, 3: Industrial, 4: Educational, 5: Others
unique_BldUse <- unique(StPaul_footprint$Bld_use)
print(unique_BldUse)

# Function to calculate concrete use
calculate_concrete_mass <- function(data) {
  data <- data %>%
    mutate(
      Concrete_Mass = case_when(
        Bld_use == 1 & YEAR_BUILT <= 1960 ~ FAm2 * `Residential (1930 - 1960)	_Concrete`,
        Bld_use == 1 & YEAR_BUILT > 1960 & YEAR_BUILT <= 1975 ~ FAm2 * `Residential (1961 - 1975)	_Concrete`,
        Bld_use == 1 & YEAR_BUILT > 1975 & YEAR_BUILT <= 1999 ~ FAm2 * `Residential (1976 - 1999)	_Concrete`,
        Bld_use == 1 & YEAR_BUILT > 1999 ~ FAm2 * `Residential (2000 - 2018)	_Concrete`,
        Bld_use == 2 ~ FAm2 * `Commercial	_Concrete`,
        Bld_use == 3 ~ FAm2 * `Industrial	_Concrete`,
        Bld_use == 4 ~ FAm2 * `Educational	_Concrete`,
        Bld_use == 0 ~ FAm2 * `Others		_Concrete`,
        TRUE ~ 0 
      )
    )
  return(data)
}

# Function to calculate cement use
calculate_cement_mass <- function(data) {
  data <- data %>%
    mutate(
      Cement_Mass = case_when(
        Bld_use == 1 & YEAR_BUILT <= 1960 ~ FAm2 * `Residential (1930 - 1960)	_Cement`,
        Bld_use == 1 & YEAR_BUILT > 1960 & YEAR_BUILT <= 1975 ~ FAm2 * `Residential (1961 - 1975)	_Cement`,
        Bld_use == 1 & YEAR_BUILT > 1975 & YEAR_BUILT <= 1999 ~ FAm2 * `Residential (1976 - 1999)	_Cement`,
        Bld_use == 1 & YEAR_BUILT > 1999 ~ FAm2 * `Residential (2000 - 2018)	_Cement`,
        Bld_use == 2 ~ FAm2 * `Commercial	_Cement`,
        Bld_use == 3 ~ FAm2 * `Industrial	_Cement`,
        Bld_use == 4 ~ FAm2 * `Educational	_Cement`,
        Bld_use == 0 ~ FAm2 * `Others		_Cement`,
        TRUE ~ 0 
      )
    )
  return(data)
}

# Apply the calculate_mass function to your data
StPaul_footprint <- calculate_concrete_mass(StPaul_footprint)
StPaul_footprint <- calculate_cement_mass(StPaul_footprint)

desired_order <- c("FID", "FAm2", "Bld_use", "FIN_SQ_FT","YEAR_BUILT", "Concrete_Mass", "Cement_Mass")
## Reorder the columns based on the desired order
StPaul_footprint <- StPaul_footprint %>% 
  select(desired_order, everything())

# Check1 <- StPaul_footprint[StPaul_footprint$Bld_use == 1 & StPaul_footprint$YEAR_BUILT<= 1960, ]
# Check2 <- StPaul_footprint[StPaul_footprint$Bld_use == 2 & StPaul_footprint$FAm2 != 0,]
# Check3 <- StPaul_footprint[StPaul_footprint$Bld_use == 3 & StPaul_footprint$FAm2 != 0,]
# Check4 <- StPaul_footprint[StPaul_footprint$Bld_use == 4 & StPaul_footprint$FAm2 != 0,]
# Check0 <- StPaul_footprint[StPaul_footprint$Bld_use == 0 & StPaul_footprint$FAm2 != 0,]

st_write(StPaul_footprint, "C:/Users/SUS/My Drive/MS Estimation/StPaul/2021/Created_Data/Material_Use_Parcel/MatarialUseParcel.shp")


# Benchmark ---------------------------------------------------------------
StPaul_footprint =  st_read("C:/Users/SUS/My Drive/MS Estimation/StPaul/2021/Created_Data/Material_Use_Parcel/MatarialUseParcel.shp")

zero_count <- sum(StPaul_footprint$FAm2 == 0)
nonzero_count <- sum(StPaul_footprint$FAm2 != 0)
nonzero_ratio = nonzero_count / nrow(StPaul_footprint)
print(nonzero_ratio)
# 84.6% of parcels have values not equal to 0


## check the percentage of each builidng useage
filtered_data <- StPaul_footprint %>%
  filter(StPaul_footprint$FAm2 != 0)
total_blds <- nrow(filtered_data)
residential_blds = sum(filtered_data$Bld_use==1)
commercial_blds = sum(filtered_data$Bld_use==2)
industrial_blds = sum(filtered_data$Bld_use==3)
educational_blds = sum(filtered_data$Bld_use==4)
others_blds = sum(filtered_data$Bld_use==0)

percentage_residential <- (residential_blds / total_blds) * 100
percentage_commercial <- (commercial_blds / total_blds) * 100
percentage_industrial <- (industrial_blds / total_blds) * 100
percentage_educational <- (industrial_blds / total_blds) * 100
percentage_others <- (others_blds / total_blds) * 100

cat("Percentage of residential buildings:", percentage_residential, "%\n")
cat("Percentage of commercial buildings:", percentage_commercial, "%\n")
cat("Percentage of inductrial buildings:", percentage_industrial, "%\n")
cat("Percentage of educational buildings:", percentage_educational, "%\n")
cat("Percentage of other type of buildings:", percentage_others, "%\n")


residential_footprint = StPaul_footprint %>%
  filter(StPaul_footprint$Bld_use == 1)
avg_BldArea = mean(residential_footprint$FAm2, na.rm=TRUE)
Total_Concrete_inCity = sum(StPaul_footprint$Cncrt_M)
Total_Cement_inCity = sum(StPaul_footprint$Cmnt_Ms)

cat("Average floor area for residential building is:", avg_BldArea, "\n")
cat("Total Concrete Use in St. Paul is:", Total_Concrete_inCity, "\n")
cat("Total Cement Use in St. Paul is:", Total_Cement_inCity, "\n")