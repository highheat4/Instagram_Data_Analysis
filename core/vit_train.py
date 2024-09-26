import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import ViTModel, ViTFeatureExtractor
from sklearn.model_selection import train_test_split
import pandas as pd
from PIL import Image
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Dataset class for loading images and log_likes
class InstagramDataset(Dataset):
    def __init__(self, data, feature_extractor):
        self.data = data
        self.feature_extractor = feature_extractor
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = self.data.iloc[idx]['image_path']
        image = Image.open(image_path).convert('RGB')  # Ensure images are RGB
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        
        # Get the log_likes for regression
        log_likes = self.data.iloc[idx]['log_likes']
        target = torch.tensor([log_likes], dtype=torch.float32)
        
        return inputs['pixel_values'].squeeze(0), target  # (C, H, W), target

# Model class that adds a regression head on top of ViT
class ViTRegressionModel(nn.Module):
    def __init__(self, pretrained_model_name='google/vit-base-patch16-224'):
        super(ViTRegressionModel, self).__init__()
        self.vit = ViTModel.from_pretrained(pretrained_model_name)
        self.regressor = nn.Sequential(
            nn.Linear(self.vit.config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)  # Single output for regression (predicting log_likes)
        )
    
    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output  # Extract CLS token embedding
        return self.regressor(pooled_output)

# Load the CSV file
data = pd.read_csv('./new_instagram_data.csv')
data['image_path'] = data['image_path'].apply(lambda x: x.replace('../Data/insta_data/', './insta_data/'))

# !Uncomment to drop log_no_comments
# data = data.drop(columns=['log_no_of_comments'])

# Split data into train and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Initialize feature extractor and datasets
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
train_dataset = InstagramDataset(train_data, feature_extractor)
test_dataset = InstagramDataset(test_data, feature_extractor)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Initialize model, loss function, and optimizer
model = ViTRegressionModel()
criterion = nn.MSELoss()  # Regression task
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for pixel_values, targets in train_loader:
        pixel_values = pixel_values.to(device)
        targets = targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(pixel_values)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')

# Save the trained model
model_save_path = './core/trained_models/fine_tuned_vit_model.pth' #! save to new location for no comments
torch.save(model.state_dict(), model_save_path)
print(f'Model saved to {model_save_path}')

# Evaluate the model on the test dataset
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for pixel_values, targets in test_loader:
        pixel_values = pixel_values.to(device)
        targets = targets.to(device)

        outputs = model(pixel_values)
        y_true.extend(targets.cpu().numpy())
        y_pred.extend(outputs.cpu().numpy())

# Convert lists to numpy arrays
y_true = np.expm1(np.array(y_true))
y_pred = np.expm1(np.array(y_pred))

# Calculate metrics
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f'Mean Squared Error: {mse:.4f}')
print(f'Mean Absolute Error: {mae:.4f}')
print(f'R-squared: {r2:.4f}')

# Plotting
plt.figure(figsize=(8, 8))
plt.scatter(y_true, y_pred, alpha=0.6)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color='red', linestyle='--')
plt.xlabel('Actual Likes')
plt.ylabel('Predicted Likes')
plt.title('Actual vs Predicted Likes for ViT; With Comments')
plt.legend(['Perfect Prediction', 'Predictions'])
plt.xlim(y_true.min(), y_true.max())
plt.ylim(y_true.min(), y_true.max())
plt.grid()
plt.show()
