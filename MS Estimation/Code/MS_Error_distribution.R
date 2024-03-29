rm(list=ls())

library(dplyr)
library(tidyr)

# Import 200 by 200 csv files
df= read.csv("/Users/akihi/Library/CloudStorage/OneDrive-PrincetonUniversity/Projects/MS Estimation/WorkFolder/MS Estimation/Code/MR_200x200.csv")

df <- df %>% filter(!is.na(Error))


df_count <- df %>%
  mutate(ErrorCategory = case_when(
    Error > 40 ~ "Greater than 40",
    Error >= 20 ~ "20-40",
    Error >= 10 ~ "10-20",
    Error >= 0 ~ "0-10",
    TRUE ~ "Other" # For handling potential NAs or negative values
  )) %>%
  group_by(ErrorCategory) %>%
  summarise(Count = n())

# View the results
print(df_count)
