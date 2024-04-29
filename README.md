# Deep Learning Approach to Floor Area and Building Material Stocks Estimation Using Aerial & Street View Image
This is the repository for the research to develop a deep learning model to predict material stocks estimation at city-scale.

The preprint is available at https://engrxiv.org/preprint/view/3604

## Work Step
Please Note that this repository does not contain all the data I used for this study because some data are too large to upload.

### Step1

Download all the necessary data. All the necessary code and information to download the data are stored in "Original_Data" folder.


### Step 2

"Spatial_Analysis_2021" folder contains code to split the NAIP image into each parcel and assign building types. 

### Step 3

Use code in "Model_Training_Evaluation" to train and test the models.

### Step 4

Use code in "MS_Estimation" to estimate material stocks (MSs) based on predicted floor area and building types and groud truth.

### Step 5

The code in "Code_For_Figures_Stats" helps you to understand the model and data better.
