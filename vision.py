import os
import pandas as pd
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torchvision.models as models

# Define a custom dataset for loading images and labels
class InstagramDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['image_path']
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        likes = self.dataframe.iloc[idx]['likes']
        return image, likes

# Image transformations (resize, normalize) - also introduces random flips and coloring to prevent overfitting
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Load the CSV and modify the paths
data = pd.read_csv('./instagram_data.csv')
data['image_path'] = data['image_path'].apply(lambda x: x.replace('../Data/insta_data/', './insta_data/'))

# Split data into training and test sets
train_data = data.sample(frac=0.8, random_state=42)
test_data = data.drop(train_data.index)

# Create datasets and data loaders
train_dataset = InstagramDataset(train_data, transform=transform)
test_dataset = InstagramDataset(test_data, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load pre-trained ResNet18 model
model = models.resnet101(pretrained=True)

# Modify the last layer for regression
num_features = model.fc.in_features
model.fc = nn.Linear(model.fc.in_features, 1)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, likes in train_loader:
        images = images.to(device)
        likes = likes.float().unsqueeze(1).to(device)  # Reshape likes to be of shape [batch_size, 1]

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, likes)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

model.eval()  # Set model to evaluation mode
y_true = []
y_pred = []

with torch.no_grad():
    for images, likes in test_loader:
        images = images.to(device)
        likes = likes.float().unsqueeze(1).to(device)
        
        outputs = model(images)
        y_true.extend(likes.cpu().numpy())
        y_pred.extend(outputs.cpu().numpy())

# Calculate performance metrics
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"Test MSE: {mse:.4f}, R2: {r2:.4f}")
