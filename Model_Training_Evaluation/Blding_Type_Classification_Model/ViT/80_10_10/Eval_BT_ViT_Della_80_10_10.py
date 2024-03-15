# Building Type Classification Model
# Evaluate trained ViT model with 80:10:10

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


## Set (hyper)parameters and specify the hardware for computation
IMG_SIZE = 224 
BATCH_SIZE = 64
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]
EPOCHS = 1000
LEARNING_RATE = 0.001

# Loading and preprocessing dataset
Grid_MR = pd.read_csv("/scratch/network/ao3526/MR_2011_2021withBldType.csv")
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
train_val_idx, test_idx, _, _ = train_test_split(np.arange(len(BldType)), BldType, stratify=BldType, test_size=0.1, random_state=42)
train_idx, val_idx, _, _ = train_test_split(train_val_idx, BldType[train_val_idx], stratify=BldType[train_val_idx], test_size=0.125, random_state=42) # 0.25 * 0.8 = 0.2

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
vit_model_path = '/scratch/network/ao3526/vit_b_16-c867db91.pth'
combined_model = ViTCombined(num_classes=num_classes, vit_model_path=vit_model_path).to(device)

test_label_encoder = test_combined_dataset.dataset.get_label_encoder()

# Iterate through the test dataset
# Load the best model weights before evaluation
best_model_path = "/scratch/network/ao3526/Results/BldType/ViT/80_10_10/Best_Combined_model.pth"
combined_model.load_state_dict(torch.load(best_model_path, map_location=device))
combined_model.eval()  # Set the model to evaluation mode
# Create empty lists to store results
ids = []
ground_truth = []
predicted_classes = []


# Iterate through the test data and make predictions
with torch.no_grad():
    for satellite_images, street_view_images, labels, img_id in test_loader:
        satellite_images, street_view_images, labels = satellite_images.to(device), street_view_images.to(device), labels.to(device)
        outputs = combined_model(satellite_images, street_view_images)

        _, predicted = torch.max(outputs, 1)  # Get the predicted class index

        ids.extend(img_id.cpu().numpy())
        ground_truth.extend(labels.cpu().numpy())
        predicted_classes.extend(predicted.cpu().numpy())

# Use the LabelEncoder to transform class indices back to original building type labels
ground_truth_labels = test_label_encoder.inverse_transform(ground_truth)
predicted_labels = test_label_encoder.inverse_transform(predicted_classes)

# Calculate the accuracy and F1 score for the test dataset
test_accuracy = accuracy_score(ground_truth, predicted_classes)
test_precision = precision_score(ground_truth, predicted_classes, average='weighted')
test_recall = recall_score(ground_truth, predicted_classes, average='weighted')
test_f1_score = f1_score(ground_truth, predicted_classes, average='weighted')

# Save the metrics to a text file
metrics_file_path = "/scratch/network/ao3526/Results/BldType/ViT/80_10_10/test_metrics.txt"
with open(metrics_file_path, "w") as metrics_file:
    metrics_file.write(f"Test Accuracy: {test_accuracy:.4f}\n")
    metrics_file.write(f"Test Precision: {test_precision:.4f}\n")
    metrics_file.write(f"Test Recall: {test_recall:.4f}\n")
    metrics_file.write(f"Test F1 Score: {test_f1_score:.4f}\n")

# Generate the confusion matrix for the test data
cm_test = confusion_matrix(ground_truth, predicted_classes)
np.save('/scratch/network/ao3526/Results/BldType/ViT/80_10_10/test_confusion_matrix.npy', cm_test)

# Plot and save the confusion matrix for the test data
plt.figure(figsize=(10, 10))
plot_confusion_matrix(cm_test, classes=test_label_encoder.classes_, title='Test Confusion Matrix')
plt.savefig('/scratch/network/ao3526/Results/BldType/ViT/80_10_10/test_confusion_matrix.png')

# Create a DataFrame with ID, ground truth, and predicted values
results_df = pd.DataFrame({"ID": ids, "Ground Truth": ground_truth_labels, "Predicted Value": predicted_labels})

# Save the DataFrame as a CSV file
results_csv_path = "/scratch/network/ao3526/Results/BldType/ViT/80_10_10/Results.csv"
results_df.to_csv(results_csv_path, index=False)