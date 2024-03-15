# This is the code to crate categorical values, representing building type (use)
rm(list=ls())


Data = read.csv("/Users/akihi/Library/CloudStorage/OneDrive-PrincetonUniversity/Projects/MS Estimation/WorkFolder/MR_2021_2011_withBldUse/MR_2011_2021withNTL.csv")

names(Data)

# Print the unique values
unique_class1 = unique(Data$USECLASS1)
print(unique_class1)

unique_dwell = unique(Data$DWELL_T)
print(unique_dwell)


# # Create subset data that only contains "Res 1 unit" in UseClass1
# Data_Res1 <- subset(Data, USECLASS1 == "Res 1 unit")
# unique_class1_Res1 = unique(Data_Res1$USECLASS1)
# unique_dwell_Res1 = unique(Data_Res1$DWELL_T)
# 
# # Create subset data that only contains "Res 2-3 unit" in UseClass1
# Data_Res2 <- subset(Data, USECLASS1 == "Res 2-3 units")
# unique_class1_Res2 = unique(Data_Res2$USECLASS1)
# unique_dwell_Res2 = unique(Data_Res2$DWELL_T)
# 
# # Create subset data that only contains "Apt 4+ units" in UseClass1
# Data_Res3 <- subset(Data, USECLASS1 == "Apt 4+ units")
# unique_class1_Res3 = unique(Data_Res3$USECLASS1)
# unique_dwell_Res3 = unique(Data_Res3$DWELL_T)

empty_count = sum(Data$DWELL_T == " ")
Data_empty = subset(Data, DWELL_T == " ")
unique_empty = unique(Data_empty$USECLASS1)

# cheking_count <- sum(Data$DWELL_T == "RESIDENTIAL, VACANT LAND, LOT")
# Data_cheking <- subset(Data, DWELL_T == "RESIDENTIAL, VACANT LAND, LOT")
# unique_cheking = unique(Data_cheking$USECLASS1)

# Create new column as categorical value for building (dwelling) type
# 0 as others (non-residential), 1 as single-family house, and 2 as multi-family house
Data$BldType = ifelse(Data$DWELL_T %in% c("RESIDENTIAL, VACANT LAND, LOT", " "), 0,
                       ifelse(Data$DWELL_T %in% c("SINGLE FAMILY DWELLING, PLATT", "SINGLE FAMILY W/ACCESSORY UNI",
                                                  "TWIN HOME", "TOWNHOME - DETACHED UNIT", "BED & BREAKFAST", "RESIDENTIAL, OTHER") , 1, 2))


Data_to_check <- Data[, c("DWELL_T", "BldType")]
View(Data_to_check)

no_of_non_residential = sum(Data$BldType == 0)
no_of_SF = sum(Data$BldType == 1)
no_of_MF = sum(Data$BldType == 2)

write.csv(Data, "/Users/akihi/Library/CloudStorage/OneDrive-PrincetonUniversity/Projects/MS Estimation/WorkFolder/MR_2021_2011_withBldUse/Created_Data/MR_2011_2021withBldType.csv", row.names = FALSE)


