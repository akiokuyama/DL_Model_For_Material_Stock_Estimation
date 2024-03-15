# Building Type Classification Model
# Train ViT model with 60:20:20

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
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from itertools import product
from sklearn.model_selection import train_test_split

from torchvision.models import vit_b_16, ViT_B_16_Weights
from torch.nn.parallel import DistributedDataParallel as DDP
from socket import gethostname
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

## Set (hyper)parameters and specify the hardware for computation
IMG_SIZE = 224
BATCH_SIZE = 64
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]
EPOCHS = 1000
LEARNING_RATE = 0.001

# Loading and preprocessing dataset
Grid_MR = pd.read_csv("/scratch/gpfs/ao3526/MR_2011_2021withBldType.csv")
Grid_MR.rename(columns={'ID': 'id'}, inplace=True)
Grid_MR['id'] = Grid_MR['id'].astype(int)

# Define image transformation rules for training and test data
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

# Data splitting for model training
BldType = Grid_MR['BldType'].values
train_val_idx, test_idx, _, _ = train_test_split(np.arange(len(BldType)), BldType, stratify=BldType, test_size=0.2, random_state=42)
train_idx, val_idx, _, _ = train_test_split(train_val_idx, BldType[train_val_idx], stratify=BldType[train_val_idx], test_size=0.25, random_state=42) # 0.25 * 0.8 = 0.2

# Download data by using custom dataset class
satellite_image_path = "/scratch/gpfs/ao3526/3Bands_new/"
cityyear = "StPaul2021"
bit = "8Bit"
street_view_image_path = "/scratch/gpfs/ao3526/GSV_Image"
class CombinedDataset(Dataset):
    def __init__(self, satellite_data, street_view_data, is_train=True): # define instance variable that is used in this class
        self.satellite_data= satellite_data
        self.street_view_data= street_view_data
        self.label_encoder = LabelEncoder()
        self.street_view_data['BldType_encoded'] = self.label_encoder.fit_transform(self.street_view_data['BldType'])
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
        label = self.street_view_data.BldType_encoded.iloc[index]
        
        return satellite_img, street_view_img, label, img_id

    def __len__(self): # return the total no of samples
        return len(self.satellite_data)
    
    def get_label_encoder(self):
        return self.label_encoder

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
    def __init__(self, num_classes, vit_model_path):
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
        
        self.classifier = nn.Sequential(
            nn.Linear(768 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)  
        )

    def forward(self, satellite_img, street_view_img):
        # Extract features from both image sources
        satellite_features = self.vit_model_satellite(satellite_img)
        street_view_features = self.vit_model_street_view(street_view_img)
        
        # Concatenate the features from both image sources
        combined_features = torch.cat((satellite_features, street_view_features), dim=1)
        
        # Classify the combined features
        output = self.classifier(combined_features)
        return output

# Initialize the model
num_classes = 3
vit_model_path = '/scratch/gpfs/ao3526/ViT/vit_b_16-c867db91.pth'
combined_model = ViTCombined(num_classes=num_classes, vit_model_path=vit_model_path).to(device)

combined_model = combined_model.to(local_rank)
combined_model = DDP(combined_model, device_ids=[local_rank])

# Convert the model to a string representation
model_str = str(combined_model)
architecture_file_path = "/scratch/gpfs/ao3526/Results/BldType/ViT/60_20_20/model_architecture.txt"

# Write the model architecture to the text file
with open(architecture_file_path, "w") as arch_file:
    arch_file.write(model_str)

# Define the loss function, optimizer, and learning rate scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(combined_model.parameters(), lr=0.0001, momentum=0.9)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

# Define early stopping parameters
early_stopping_patience = 20 
best_val_loss = float('inf') 
epochs_without_improvement = 0

# Function to calculate confusion matrix
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks= np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max()/2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment = 'center',
                 color = 'white' if cm[i, j] > thresh else 'black')
        
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout


# Training the model
with open("/scratch/gpfs/ao3526/Results/BldType/ViT/60_20_20/loss_log.txt", "w") as log_file:
    for epoch in range(EPOCHS):
        combined_model.train()
        running_train_loss = 0.0
        all_train_labels = []
        all_train_predictions = []
        train_losses = []
        val_losses = []

        # Training loop
        for satellite_images, street_view_images, labels, _ in train_loader:
            satellite_images, street_view_images, labels = satellite_images.to(device), street_view_images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = combined_model(satellite_images, street_view_images)
            loss = criterion(outputs, labels) 
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * satellite_images.size(0)

            predicted_classes = torch.argmax(outputs, dim=1)
            all_train_labels.extend(labels.cpu().numpy()) 
            all_train_predictions.extend(predicted_classes.cpu().numpy()) 

        train_loss = running_train_loss / len(train_loader.dataset)
        train_accuracy = accuracy_score(all_train_labels, all_train_predictions)
        train_f1 = f1_score(all_train_labels, all_train_predictions, average = 'weighted') 
        train_losses.append(train_loss)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train F1: {train_f1:.4f}")

        # Validation loop
        combined_model.eval()
        running_val_loss = 0.0
        all_val_labels = []
        all_val_predictions = []

        with torch.no_grad():
            for satellite_images, street_view_images, labels, _ in validation_loader:
                satellite_images, street_view_images, labels = satellite_images.to(device), street_view_images.to(device), labels.to(device)

                outputs = combined_model(satellite_images, street_view_images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * satellite_images.size(0)

                predicted_classes = torch.argmax(outputs, dim=1)
                all_val_labels.extend(labels.cpu().numpy())
                all_val_predictions.extend(predicted_classes.cpu().numpy())

        val_loss = running_val_loss / len(validation_loader.dataset)
        val_accuracy = accuracy_score(all_val_labels, all_val_predictions)
        val_f1 = f1_score(all_val_labels, all_val_predictions, average='weighted')
        val_losses.append(val_loss)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation F1: {val_f1:.4f}")
        
        # Write to log file
        log_file.write(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train F1: {train_f1:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation F1: {val_f1:.4f}\n")
        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(combined_model.state_dict(), "/scratch/gpfs/ao3526/Results/BldType/ViT/60_20_20/Best_Combined_model.pth")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping after {epoch+1} epochs without improvement.")
            break



    # Optionally, plot training and validation loss over epochs
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss')  # Plot training loss over epochs
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')    # Plot validation loss over epochs
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.savefig("/scratch/gpfs/ao3526/Results/BldType/ViT/60_20_20/Combined_Model.png")
    plt.show()

dist.destroy_process_group()

# Confusion Matrix for Training and Test Data
cm_train = confusion_matrix(all_train_labels, all_train_predictions)
np.save("/scratch/gpfs/ao3526/Results/BldType/ViT/60_20_20/train_confusion_matrix.npy", cm_train)
cm_val = confusion_matrix(all_val_labels, all_val_predictions)
np.save('/scratch/gpfs/ao3526/Results/BldType/ViT/60_20_20/test_confusion_matrix.npy', cm_val)  # Saving the confusion matrix

# Encode Labels
train_label_encoder = train_combined_dataset.dataset.get_label_encoder()
validation_label_encoder = validation_combined_dataset.dataset.get_label_encoder()

# Plot the confusion matrix
plt.figure(figsize=(10, 10))
plot_confusion_matrix(cm_train, classes=train_label_encoder.classes_, title='Training Confusion Matrix')
plt.savefig('/scratch/gpfs/ao3526/Results/BldType/ViT/60_20_20/train_confusion_matrix.png')

plt.figure(figsize=(10, 10))
plot_confusion_matrix(cm_val, classes=validation_label_encoder.classes_, title='Validation Confusion Matrix')
plt.savefig('/scratch/gpfs/ao3526/Results/BldType/ViT/60_20_20/validation_confusion_matrix.png')