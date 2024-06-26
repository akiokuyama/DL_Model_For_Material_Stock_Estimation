# Floor Area Prediction Model
# Evaluate trained ViT model with 60:20:20

# Import necessary libraries
import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader, Dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

import matplotlib.pyplot as plt
from PIL import Image
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from itertools import product
from sklearn.model_selection import train_test_split
from torchvision.models import vit_b_16, ViT_B_16_Weights

from torch.nn.parallel import DistributedDataParallel as DDP
from socket import gethostname
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


# Set (hyper)parameters and specify the hardware for computation
IMG_SIZE = 224
BATCH_SIZE = 64 
IMG_MEAN = [0.485, 0.456, 0.406] 
IMG_STD = [0.229, 0.224, 0.225] 
EPOCHS = 1000
LEARNING_RATE = 0.001

# Load and prepare the dataset
Grid_MR = pd.read_csv("/scratch/network/ao3526/MR_2011_2021withBldType.csv")
Grid_MR.rename(columns={'ID': 'id'}, inplace=True)
Grid_MR.rename(columns={'FA_chng': 'FA_Change'}, inplace=True)
Grid_MR.rename(columns={'FA22021': 'FA2021'}, inplace=True)
Grid_MR['id'] = Grid_MR['id'].astype(int)
FA2021 = Grid_MR['FA2021'].values
BldType = Grid_MR['BldType'].values

# Starndarlization
features = Grid_MR[['FA2021']]
scaler = RobustScaler()
scaled_features = scaler.fit_transform(features)
# Replace original values with standardized values
Grid_MR[['FA2021']] = scaled_features

# Store the mean and std dev for 'FA' after standardization. I need these values to reverse FloorArea
fa_median = scaler.center_[0] 
fa_iqr = scaler.scale_[0]

# Define image transformation rules for training and validation data
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(3),
    transforms.ToTensor(),
    transforms.Normalize(IMG_MEAN, IMG_STD)
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMG_MEAN, IMG_STD)
])

# Split the dataset into training, validation, and test sets
train_val_idx, test_idx, _, _ = train_test_split(np.arange(len(BldType)), BldType, stratify=BldType, test_size=0.05, random_state=42)
train_idx, val_idx, _, _ = train_test_split(train_val_idx, BldType[train_val_idx], stratify=BldType[train_val_idx], test_size=0.0526, random_state=42)

# Download data by using custom dataset class
satellite_image_path = "/scratch/network/ao3526/3Bands_new_Sharpen/"
cityyear = "StPaul2021"
bit = "8Bit"
street_view_image_path = "/scratch/network/ao3526/GSV_Image"
class CombinedDataset(Dataset):
    def __init__(self, satellite_data, street_view_data, is_train=True): # define instance variable that is used in this class
        self.satellite_data= satellite_data
        self.street_view_data= street_view_data
        if is_train: # check if it is training or validation data. Process the transform
            self.transforms = train_transforms
        else:
            self.transforms = val_transforms

    def __getitem__(self, index): # return single data item at the specified index
        # street view image
        street_view_image_name = os.path.join(street_view_image_path, f"GSV_{index}.jpg") # Construct the path for the "north" image
        
        if os.path.exists(street_view_image_name):
            street_view_img = Image.open(street_view_image_name) # open the image
        else:
            street_view_img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=(128, 128, 128)) # Assign gray images for the ID without images
        
        street_view_img = self.transforms(street_view_img)
        img_id = self.street_view_data.id.iloc[index] # get the image ID
        
        # satellite image
        satellite_image_name = os.path.join(satellite_image_path, f"{cityyear}_{index}_{bit}_3bands.tif") # construct the path for the image
        satellite_img = Image.open(satellite_image_name)
        satellite_img = self.transforms(satellite_img) # apply the data transformation
        
        # other data
        label = self.street_view_data.FA2021.iloc[index]

        return satellite_img, street_view_img, label, img_id

    def __len__(self): # return the total no of samples
        return len(self.satellite_data)

# Creating dataset instances
test_combined_dataset = Subset(CombinedDataset(Grid_MR, Grid_MR, is_train=False), test_idx)


# DataLoader setup for cluster computing. Use parallel processing
num_workers = int(os.environ["SLURM_CPUS_PER_TASK"]) # You can change this number as needed
# Check for CUDA availability
use_cuda = torch.cuda.is_available()
# DataLoader arguments
test_kwargs = {'batch_size': 1, 'shuffle': False, 'num_workers': num_workers}
if use_cuda:
    cuda_kwargs = {'pin_memory': True}
    test_kwargs.update(cuda_kwargs)

# Create dataloader test set
test_loader = DataLoader(dataset=test_combined_dataset, **test_kwargs)

# Check if code correctly set the no. of GPUs
print("Num GPUs Available: ", torch.cuda.device_count())

# Set device and optimizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# Define the model architecture
class ViTCombined(nn.Module):
    def __init__(self, vit_model_path):
        super(ViTCombined, self).__init__()
        # Load the pre-trained ViT model
        self.vit_model_satellite = vit_b_16(weights=None)
        self.vit_model_street_view = vit_b_16(weights=None)
        
        # Load the state dict for both ViT models from the provided path
        state_dict = torch.load(vit_model_path)
        self.vit_model_satellite.load_state_dict(state_dict)
        self.vit_model_street_view.load_state_dict(state_dict)
        
        # Remove the classification head of both ViT models to use them as feature extractors
        self.vit_model_satellite.heads = nn.Identity()
        self.vit_model_street_view.heads = nn.Identity()
        
        self.regression = nn.Sequential(
            nn.Linear(768 * 2, 512),  # Combine features from both models
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)  # Output layer for the number of classes
        )

    def forward(self, satellite_img, street_view_img):
        # Extract features from both image sources
        satellite_features = self.vit_model_satellite(satellite_img)
        street_view_features = self.vit_model_street_view(street_view_img)
        
        # Concatenate the features from both image sources
        combined_features = torch.cat((satellite_features, street_view_features), dim=1)
        
        # Classify the combined features
        output = self.regression(combined_features)
        return output

# Initialize the model
vit_model_path = '/scratch/network/ao3526/ViT/vit_b_16-c867db91.pth'
combined_model = ViTCombined(vit_model_path=vit_model_path).to(device)

# Iterate through the test dataset
best_model_path = "/scratch/network/ao3526/Results/FA2021/ViT/90_5_5/Best_Stratified_Sample.pth"
combined_model.load_state_dict(torch.load(best_model_path, map_location=device), strict=False)
combined_model.eval()  # Set the model to evaluation mode

# Create empty lists to store results
ids = []
ground_truth = []
predicted_values = []
ground_truth_unscaled = []
predicted_values_unscaled = []

# Iterate through the test data and make predictions
with torch.no_grad():
    for satellite_images, street_view_images, labels, img_id in test_loader:
        satellite_images, street_view_images, labels = satellite_images.to(device), street_view_images.to(device), labels.to(device)
        outputs = combined_model(satellite_images, street_view_images)
        prediction = outputs.item()
        predicted_unscaled = prediction * fa_iqr + fa_median

        # Correctly access ground truth from Grid_MR using img_id
        ground_truth_value = Grid_MR.loc[Grid_MR['id'] == img_id.item(), 'FA2021'].values[0]
        
        ids.append(img_id.item())
        ground_truth.append(ground_truth_value)
        predicted_values.append(prediction)
        ground_truth_unscaled_value = ground_truth_value * fa_iqr + fa_median
        ground_truth_unscaled.append(ground_truth_unscaled_value)
        predicted_values_unscaled.append(predicted_unscaled)

# ground_truth_unscaled and predicted_values_unscaled are lists of actual and predicted values, respectively
ground_truth_unscaled = np.array(ground_truth_unscaled)
predicted_values_unscaled = np.array(predicted_values_unscaled)

# Calculate the accuracy metrics for test dataset based on unscaled values
mse = mean_squared_error(ground_truth_unscaled, predicted_values_unscaled)
rmse = np.sqrt(mse)
mae = mean_absolute_error(ground_truth_unscaled, predicted_values_unscaled)
r2 = r2_score(ground_truth_unscaled, predicted_values_unscaled)

# Calculate the accuracy metrics for test dataset based on scaled values
mse_scaled = mean_squared_error(ground_truth, predicted_values)
rmse_scaled = np.sqrt(mse_scaled)
mae_scaled = mean_absolute_error(ground_truth, predicted_values)
ground_truth_array = np.array(ground_truth)
predicted_values_array = np.array(predicted_values)
r2_scaled = r2_score(ground_truth, predicted_values)

# Calculate total error
sum_groud_truth = sum(ground_truth_unscaled)
sum_predicted_value = sum(predicted_values_unscaled)
Total_Error = np.abs((sum_predicted_value - sum_groud_truth)/sum_groud_truth) * 100
print(f"MSE: {mse}, RMSE: {rmse}, MAE: {mae}, R²: {r2}, MSE_scaled: {mse_scaled}, RMSE_scaled: {rmse_scaled}, MAE_scaled: {mae_scaled}, R²_scaled: {r2_scaled}, Total Error: {Total_Error}")

# Define the path for the MSE loss text file
mse_loss_file_path = "/scratch/network/ao3526/Results/FA2021/ViT/90_5_5/MSE_Loss_TestSet.txt"

# Write the MSE loss to the text file
with open(mse_loss_file_path, "w") as file:
    file.write(f"MSE: {mse}, RMSE: {rmse}, MAE: {mae}, R²: {r2}, MSE_scaled: {mse_scaled}, RMSE_scaled: {rmse_scaled}, MAE_scaled: {mae_scaled}, R²_scaled: {r2_scaled}, Total Error: {Total_Error}")
            
# Create a DataFrame with ID, ground truth, and predicted values
results_df = pd.DataFrame({"ID": ids, "Ground Truth": ground_truth, "Predicted Value": predicted_values, "Ground Truth Unscaled": ground_truth_unscaled, "Predicted Value Unsclaed": predicted_values_unscaled})

# Save the DataFrame as a CSV file
results_csv_path = "/scratch/network/ao3526/Results/FA2021/ViT/90_5_5/TestSet_Results.csv"
results_df.to_csv(results_csv_path, index=False)