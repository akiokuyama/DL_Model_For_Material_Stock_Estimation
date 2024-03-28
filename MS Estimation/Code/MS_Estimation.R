rm(list=ls())

library(dplyr)
library(tidyr)

# Import Merged csv files
Merged_data = read.csv("/Users/akihi/Library/CloudStorage/OneDrive-PrincetonUniversity/Projects/MS Estimation/Results/MS_Estimation/Merged.csv")

# Import Material Intensity
MI_data = read.csv("/Users/akihi/Library/CloudStorage/GoogleDrive-ao3526@princeton.edu/My Drive/MS Estimation/StPaul/Original_Data/Material_Intensity/Material_Intensity.csv")

Material_List = c("Concrete", "Cement", "Wood", "Brick", "Gypsum", "Aggregates", "Asphalt", "Steel", "Total")


# Initialize empty lists to store the variables for each building type
SF <- list() # Single Family
MF <- list() # Multiple Family
NonResi <- list() # Non-Residential

# Loop through each material and assign the corresponding values
for(material in Material_List) {
  SF[[material]] <- MI_data[[material]][1]
  MF[[material]] <- MI_data[[material]][2]
  NonResi[[material]] <- MI_data[[material]][3]
}

# Adjusted function to get the material variable based on BldType_True
get_material_variable <- function(material, bld_type, SF, MF, NonResi) {
  if(bld_type == 1) {
    return(SF[[material]])
  } else if(bld_type == 2) {
    return(MF[[material]])
  } else if(bld_type == 0) {
    return(NonResi[[material]])
  } else {
    return(NA_real_) # default case for unexpected bld_type values
  }
}

# Loop to create True and Pred columns for each material in Material_List
for(material in Material_List) {
  # Create True column
  Merged_data <- Merged_data %>%
    mutate(!!paste0(material, "_True") := FA_True * mapply(function(bld_type) get_material_variable(material, bld_type, SF, MF, NonResi), BldType_True))
  
  # Create Pred column
  Merged_data <- Merged_data %>%
    mutate(!!paste0(material, "_Pred") := FA_Pred * mapply(function(bld_type) get_material_variable(material, bld_type, SF, MF, NonResi), BldType_True))
}

# Delete columns for scaled FA values
Merged_data$FA_True_scaled <- NULL
Merged_data$FA_Pred_scaled <- NULL

# Save the Results
write.csv(Merged_data, "/Users/akihi/Library/CloudStorage/OneDrive-PrincetonUniversity/Projects/MS Estimation/Results/MS_Results.csv", row.names = FALSE)

# Sum of MS
column_sums <- colSums(Merged_data[-(1:4)], na.rm = TRUE)

# Convert the sums to a dataframe
Sum_MS <- as.data.frame(t(column_sums), stringsAsFactors = FALSE)
# kg to 1000ton
Sum_MS <- Sum_MS %>%
  mutate(across(c(FA_True:Steel_Pred, Total_True:Total_Pred), ~ . / 1000000))


# Calculate Error as a new function
calculate_error <- function(truth, pred) {
  abs(truth - pred) / truth * 100
}

# Pivot the data to long format
long_MS <- Sum_MS %>%
  pivot_longer(cols = everything(), names_to = "Material_Type", values_to = "Value")

# Separate the material from its True/Pred status
long_MS <- long_MS %>%
  separate(Material_Type, into = c("Material", "Type"), sep = "_")

# Create a wide format with Ground Truth and Prediction
wide_MS <- long_MS %>%
  pivot_wider(names_from = "Type", values_from = "Value")

# Calculate the Error
wide_MS <- wide_MS %>%
  mutate(Error = calculate_error(True, Pred))

final_table <- wide_MS %>%
  pivot_longer(cols = c(True, Pred, Error), names_to = "Category", values_to = "Value") %>%
  pivot_wider(names_from = "Material", values_from = "Value", names_sort = FALSE)


write.csv(final_table, "/Users/akihi/Library/CloudStorage/OneDrive-PrincetonUniversity/Projects/MS Estimation/Results/MS_Sum_Results.csv", row.names = FALSE)


## Subcategory Calculation
rm(list=ls())

library(dplyr)
library(tidyr)


MS_data = read.csv("/Users/akihi/Library/CloudStorage/OneDrive-PrincetonUniversity/Projects/MS Estimation/Results/MS_Estimation/MS_Results.csv")

# Truth is 0 (Non-residential)
MS_data_Non_resi <- subset(MS_data, BldType_True == 0)

# Truth is 1 (SFHs)
MS_data_SFH <- subset(MS_data, BldType_True == 1)

# Truth is 2 (MFHs)
MS_data_MFH <- subset(MS_data, BldType_True == 2)

# Random selection
set.seed(42) # You can use any positive integer here
MS_data_random = sample_n(MS_data, size = 100)

# Create function to calculate sum
process_MS_data <- function(Merged_data) {
  # Sum of MS
  column_sums <- colSums(Merged_data[-(1:4)], na.rm = TRUE)
  
  # Convert the sums to a dataframe
  Sum_MS <- as.data.frame(t(column_sums), stringsAsFactors = FALSE)
  # kg to 1000ton
  Sum_MS <- Sum_MS %>%
    mutate(across(c(FA_True:Steel_Pred, Total_True:Total_Pred), ~ . / 1000000))
  
  
  # Calculate Error as a new function
  calculate_error <- function(truth, pred) {
    abs(truth - pred) / truth * 100
  }
  
  # Pivot the data to long format
  long_MS <- Sum_MS %>%
    pivot_longer(cols = everything(), names_to = "Material_Type", values_to = "Value")
  
  # Separate the material from its True/Pred status
  long_MS <- long_MS %>%
    separate(Material_Type, into = c("Material", "Type"), sep = "_")
  
  # Create a wide format with Ground Truth and Prediction
  wide_MS <- long_MS %>%
    pivot_wider(names_from = "Type", values_from = "Value")
  
  # Calculate the Error
  wide_MS <- wide_MS %>%
    mutate(Error = calculate_error(True, Pred))
  
  final_table <- wide_MS %>%
    pivot_longer(cols = c(True, Pred, Error), names_to = "Category", values_to = "Value") %>%
    pivot_wider(names_from = "Material", values_from = "Value", names_sort = FALSE)

return (final_table)
}

Aggregated_MS_Non_Resi = process_MS_data(MS_data_Non_resi)
Aggregated_MS_SFH = process_MS_data(MS_data_SFH)
Aggregated_MS_MFH = process_MS_data(MS_data_MFH)
Aggregated_MS_random = process_MS_data(MS_data_random)

write.csv(Aggregated_MS_Non_Resi, "/Users/akihi/Library/CloudStorage/OneDrive-PrincetonUniversity/Projects/MS Estimation/Results/MS_Estimation/MS_Non_Resi.csv", row.names = FALSE)
write.csv(Aggregated_MS_SFH, "/Users/akihi/Library/CloudStorage/OneDrive-PrincetonUniversity/Projects/MS Estimation/Results/MS_Estimation/MS_SFH.csv", row.names = FALSE)
write.csv(Aggregated_MS_MFH, "/Users/akihi/Library/CloudStorage/OneDrive-PrincetonUniversity/Projects/MS Estimation/Results/MS_Estimation/MS_MFH.csv", row.names = FALSE)
write.csv(Aggregated_MS_random, "/Users/akihi/Library/CloudStorage/OneDrive-PrincetonUniversity/Projects/MS Estimation/Results/MS_Estimation/MS_ramdom.csv", row.names = FALSE)

## Aggregated_MS_Non_Resi had very bad accuracy because most of the FA in this subset is 0.
# summary(MS_data_Non_resi$FA_True)
# zero_count <- sum(MS_data_Non_resi$FA_True == 0, na.rm = TRUE)

## Cheking the distribution of random subset
# NonResi_count <- sum(MS_data_random$BldType_True == 0, na.rm = TRUE) # 10
# SFH_count <- sum(MS_data_random$BldType_True == 1, na.rm = TRUE) # 67
# MFH_count <- sum(MS_data_random$BldType_True == 2, na.rm = TRUE) # 23
