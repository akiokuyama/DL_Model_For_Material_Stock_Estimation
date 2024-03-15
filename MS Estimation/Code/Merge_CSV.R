rm(list=ls())

library(dplyr)

# Import two csv files
BldType_csv = read.csv("/Users/akihi/Library/CloudStorage/OneDrive-PrincetonUniversity/Projects/MS Estimation/Results/BldType/CNN_ResNet18/60_20_20/Results.csv")
FA_csv = read.csv("/Users/akihi/Library/CloudStorage/OneDrive-PrincetonUniversity/Projects/MS Estimation/Results/FloorArea/CNN_ResNet18/Results/60_20_20/ValidationSet_Results.csv")

# Merge csv files
Merged_csv = merge(BldType_csv, FA_csv, by = 'ID')

Merged_csv = Merged_csv %>%
  rename('BldType_True' = 'Ground.Truth.x', 
         'BldType_Pred' = 'Predicted.Value.x',
         'FA_True_scaled' = 'Ground.Truth.y',
         'FA_Pred_scaled' = 'Predicted.Value.y',
         'FA_True' = 'Ground.Truth.Unscaled',
         'FA_Pred' = 'Predicted.Value.Unsclaed')

sum_True_FA = sum(Merged_csv$FA_True)
sum_Pred_FA = sum(Merged_csv$FA_Pred)
Total_Error = (abs(sum_True_FA - sum_Pred_FA)/sum_True_FA) * 100


# Write the data
write.csv(Merged_csv, "/Users/akihi/Library/CloudStorage/OneDrive-PrincetonUniversity/Projects/MS Estimation/Results/Merged.csv")

