# Floor Area Prediction Model
# Train the ViT model with 90:5:5 split ratio

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
train_combined_dataset = Subset(CombinedDataset(Grid_MR, Grid_MR, is_train=True), train_idx)  # Pass Grid_MR as both satellite and street view data
validation_combined_dataset = Subset(CombinedDataset(Grid_MR, Grid_MR, is_train=False), val_idx)

# DataLoader setup for cluster computing. Use parallel processing
rank          = int(os.environ["SLURM_PROCID"])
world_size    = int(os.environ["WORLD_SIZE"])
gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
print(torch.cuda.device_count())

assert gpus_per_node == torch.cuda.device_count()
print(f"Hello from rank {rank} of {world_size} on {gethostname()} where there are" \
      f" {gpus_per_node} allocated GPUs per node.", flush=True)

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
setup(rank, world_size)
if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)

local_rank = rank - gpus_per_node * (rank // gpus_per_node)
torch.cuda.set_device(local_rank)
print(f"host: {gethostname()}, rank: {rank}, local_rank: {local_rank}")


# Create sampler for both training and test sets
train_sampler = DistributedSampler(train_combined_dataset)
validation_sampler = DistributedSampler(validation_combined_dataset)


# Set the number of worker threads for DataLoader
num_workers = int(os.environ["SLURM_CPUS_PER_TASK"]) # You can change this number as needed

# Check for CUDA availability
use_cuda = torch.cuda.is_available()

# Create dataloader for both training and validation sets
train_loader = DataLoader(dataset=train_combined_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=num_workers)
validation_loader = DataLoader(dataset=validation_combined_dataset, batch_size=1, sampler=validation_sampler, num_workers=num_workers)

# Check if code correctly set the no. of GPUs
print("Num GPUs Available: ", torch.cuda.device_count())

# Set device and optimizer
device = torch.device(f"cuda:{local_rank}")


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
vit_model_path = '/scratch/gpfs/ao3526/ViT/vit_b_16-c867db91.pth'
combined_model = ViTCombined(vit_model_path=vit_model_path).to(device)

combined_model = combined_model.to(local_rank)
combined_model = DDP(combined_model, device_ids=[local_rank])

# Convert the model to a string representation
model_str = str(combined_model)
architecture_file_path = "/scratch/gpfs/ao3526/Results/BldType/ViT/90_5_5/model_architecture.txt"

# Write the model architecture to the text file
with open(architecture_file_path, "w") as arch_file:
    arch_file.write(model_str)

# Define the loss function, optimizer, and learning rate scheduler
criterion = nn.MSELoss()
optimizer = optim.SGD(combined_model.parameters(), lr=0.0001, momentum=0.9)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

# Define early stopping parameters
early_stopping_patience = 20  # Number of epochs to wait before stopping if validation loss doesn't improve
best_val_loss = float('inf')  # Initialize the best validation loss to positive infinity
epochs_without_improvement = 0  # Initialize the count of epochs without improvement

# Training the model
with open("/scratch/gpfs/ao3526/Results/FA/ViT/90_5_5/loss_log.txt", "w") as log_file:
    for epoch in range(EPOCHS):
        train_losses = []
        val_losses = []
        train_sampler.set_epoch(epoch)
        combined_model.train()
        running_train_loss = 0.0
        running_val_loss = 0.0

        # Training loop
        for satellite_images, street_view_images, labels, _ in train_loader:  # Retrieve images and labels from the data loader
            satellite_images = satellite_images.to(device)
            street_view_images = street_view_images.to(device)
            labels = labels.to(device).float()  # Cast labels to float

            optimizer.zero_grad()

            outputs = combined_model(satellite_images, street_view_images)
            loss = criterion(outputs.view(-1), labels) # view(-1) is used to flatten tensor
            
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * satellite_images.size(0)  # Use satellite_images.size(0)

        train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        print(f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {train_loss:.4f}")

        # Validation loop
        combined_model.eval()
        with torch.no_grad():
            for satellite_images, street_view_images, labels, _ in validation_loader:
                satellite_images = satellite_images.to(device)
                street_view_images = street_view_images.to(device)
                labels = labels.to(device).float()  # Cast labels to float

                outputs = combined_model(satellite_images, street_view_images)
                loss = criterion(outputs.view(-1), labels)
                running_val_loss += loss.item() * satellite_images.size(0)

        val_loss = running_val_loss / len(validation_loader.dataset)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{EPOCHS}, Validation Loss: {val_loss:.4f}")

        log_file.write(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}\n")

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # Save the model checkpoint if validation loss improves
            torch.save(combined_model.state_dict(), "/scratch/gpfs/ao3526/Results/FA/ViT/90_5_5/Best_Stratified_Sample.pth")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping after {epoch+1} epochs without improvement.")
            break  # Stop training


    # Optionally, plot training and validation loss over epochs
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.savefig("/scratch/gpfs/ao3526/Results/FA/ViT/90_5_5/BasedOnBldType.png")
    plt.show()

dist.destroy_process_group()