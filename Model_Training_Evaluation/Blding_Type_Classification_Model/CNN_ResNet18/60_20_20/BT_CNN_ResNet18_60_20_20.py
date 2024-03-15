# Building Type Classification Model
# CNN model with 60:20:20 split 

# Import necessary libraries
import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader, Dataset
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

import seaborn as sns
from itertools import product
from sklearn.model_selection import train_test_split


# Set (hyper)parameters and specify the hardware for computation
IMG_SIZE = 224
BATCH_SIZE = 64
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]
EPOCHS = 100
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
train_val_idx, test_idx, _, _ = train_test_split(np.arange(len(BldType)), BldType, stratify=BldType, test_size=0.2, random_state=42)
train_idx, val_idx, _, _ = train_test_split(train_val_idx, BldType[train_val_idx], stratify=BldType[train_val_idx], test_size=0.25, random_state=42) # 0.25 * 0.8 = 0.2

# Download data by using custom dataset class
satellite_image_path = "/scratch/network/ao3526/3Bands_new/"
cityyear = "StPaul2021"
bit = "8Bit"
street_view_image_path = "/scratch/network/ao3526/GSV_Image"
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
test_combined_dataset = Subset(CombinedDataset(Grid_MR, Grid_MR, is_train=False), test_idx)

# DataLoader setup for cluster computing
num_workers = 8  
use_cuda = torch.cuda.is_available()
train_kwargs = {'batch_size': BATCH_SIZE, 'shuffle': True, 'num_workers': num_workers}
validation_kwargs = {'batch_size': 1, 'shuffle': False, 'num_workers': num_workers}
test_kwargs = {'batch_size': 1, 'shuffle': False, 'num_workers': num_workers}

if use_cuda:
    cuda_kwargs = {'pin_memory': True}
    train_kwargs.update(cuda_kwargs)
    validation_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

# Create dataloader for both training and test sets
train_loader = DataLoader(dataset=train_combined_dataset, **train_kwargs) # Shuffling the data during training helps introduce randomness and prevents the model from overfitting to a specific order of samples
validation_loader = DataLoader(dataset=validation_combined_dataset, **validation_kwargs)
test_loader = DataLoader(dataset=test_combined_dataset, **test_kwargs)

# Set device and optimizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model definition
class CombinedModel(nn.Module):
    def __init__(self, pretrained_model, num_classes=3):
        super(CombinedModel, self).__init__()
        num_features = pretrained_model.fc.in_features
        self.cnn = nn.Sequential(*list(pretrained_model.children())[:-1])
        self.fc = nn.Linear(num_features * 2, num_classes)
    
    def forward(self, satellite_img, street_view_img):
        cnn_features_satellite = self.cnn(satellite_img).view(satellite_img.size(0), -1)
        cnn_features_street_view = self.cnn(street_view_img).view(street_view_img.size(0), -1)  # Fix this line
        combined_features = torch.cat([cnn_features_satellite, cnn_features_street_view], dim=1).float()    
        output = self.fc(combined_features)
        return output
    
# Load the local weights into the model
pretrained_model = models.resnet18()
state_dict = torch.load("/scratch/network/ao3526/ResNet18/resnet18.pth")
pretrained_model.load_state_dict(state_dict)

# Create the combined model
combined_model = CombinedModel(pretrained_model)
combined_model.to(device).float()

# Wrap the model with DataParallel
if torch.cuda.device_count() > 1:
    combined_model = nn.DataParallel(combined_model)

# Convert the model to a string representation
model_str = str(combined_model)
architecture_file_path = "/scratch/network/ao3526/Results/BldType/CNN_ResNet18/60_20_20/model_architecture.txt"

# Write the model architecture to the text file
with open(architecture_file_path, "w") as arch_file:
    arch_file.write(model_str)

# Define the loss function, optimizer, and learning rate scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(combined_model.parameters(), lr=0.0001, momentum=0.9)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

# Define early stopping parameters
early_stopping_patience = 20  # Number of epochs to wait before stopping if validation loss doesn't improve
best_val_loss = float('inf')  # Initialize the best validation loss to positive infinity
epochs_without_improvement = 0  # Initialize the count of epochs without improvement


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
with open("/scratch/network/ao3526/Results/BldType/CNN_ResNet18/60_20_20/loss_log.txt", "w") as log_file:
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
            loss = criterion(outputs, labels)  # Make sure criterion is CrossEntropyLoss
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * satellite_images.size(0)

            predicted_classes = torch.argmax(outputs, dim=1)
            all_train_labels.extend(labels.cpu().numpy()) # list of all train labels
            all_train_predictions.extend(predicted_classes.cpu().numpy()) # list of all correct predictions

        train_loss = running_train_loss / len(train_loader.dataset)
        train_accuracy = accuracy_score(all_train_labels, all_train_predictions)
        train_f1 = f1_score(all_train_labels, all_train_predictions, average = 'weighted') # weighted average for multi-class
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
            torch.save(combined_model.state_dict(), "/scratch/network/ao3526/Results/BldType/CNN_ResNet18/60_20_20/Best_Combined_model.pth")
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
    plt.savefig("/scratch/network/ao3526/Results/BldType/CNN_ResNet18/60_20_20/Combined_Model.png")
    plt.show()


# Confusion Matrix for Training and Validation Data
cm_train = confusion_matrix(all_train_labels, all_train_predictions)
np.save("/scratch/network/ao3526/Results/BldType/CNN_ResNet18/60_20_20/train_confusion_matrix.npy", cm_train)
cm_val = confusion_matrix(all_val_labels, all_val_predictions)
np.save('/scratch/network/ao3526/Results/BldType/CNN_ResNet18/60_20_20/test_confusion_matrix.npy', cm_val)  # Saving the confusion matrix

# Encode Labels
train_label_encoder = train_combined_dataset.dataset.get_label_encoder()
validation_label_encoder = validation_combined_dataset.dataset.get_label_encoder()
test_label_encoder = test_combined_dataset.dataset.get_label_encoder()

# Plot the confusion matrix
plt.figure(figsize=(10, 10))
plot_confusion_matrix(cm_train, classes=train_label_encoder.classes_, title='Training Confusion Matrix')
plt.savefig('/scratch/network/ao3526/Results/BldType/CNN_ResNet18/60_20_20/train_confusion_matrix.png')

plt.figure(figsize=(10, 10))
plot_confusion_matrix(cm_val, classes=validation_label_encoder.classes_, title='Validation Confusion Matrix')
plt.savefig('/scratch/network/ao3526/Results/BldType/CNN_ResNet18/60_20_20/validation_confusion_matrix.png')


# Load the best model weights before evaluation
best_model_path = "/scratch/network/ao3526/Results/BldType/CNN_ResNet18/60_20_20/Best_Combined_model.pth"
combined_model.load_state_dict(torch.load(best_model_path))
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
metrics_file_path = "/scratch/network/ao3526/Results/BldType/CNN_ResNet18/60_20_20/test_metrics.txt"
with open(metrics_file_path, "w") as metrics_file:
    metrics_file.write(f"Test Accuracy: {test_accuracy:.4f}\n")
    metrics_file.write(f"Test Precision: {test_precision:.4f}\n")
    metrics_file.write(f"Test Recall: {test_recall:.4f}\n")
    metrics_file.write(f"Test F1 Score: {test_f1_score:.4f}\n")

# Generate the confusion matrix for the test data
cm_test = confusion_matrix(ground_truth, predicted_classes)
np.save('/scratch/network/ao3526/Results/BldType/CNN_ResNet18/60_20_20/test_confusion_matrix.npy', cm_test)

# Plot and save the confusion matrix for the test data
plt.figure(figsize=(10, 10))
plot_confusion_matrix(cm_test, classes=test_label_encoder.classes_, title='Test Confusion Matrix')
plt.savefig('/scratch/network/ao3526/Results/BldType/CNN_ResNet18/60_20_20/test_confusion_matrix.png')

# Create a DataFrame with ID, ground truth, and predicted values
results_df = pd.DataFrame({"ID": ids, "Ground Truth": ground_truth_labels, "Predicted Value": predicted_labels})

# Save the DataFrame as a CSV file
results_csv_path = "/scratch/network/ao3526/Results/BldType/CNN_ResNet18/60_20_20/Results.csv"
results_df.to_csv(results_csv_path, index=False)