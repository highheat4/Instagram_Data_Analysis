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

# Dataset class (unchanged)
class InstagramDataset(Dataset):
    def __init__(self, data, feature_extractor):
        self.data = data
        self.feature_extractor = feature_extractor
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = self.data.iloc[idx]['image_path']
        image = Image.open(image_path).convert('RGB')
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        
        log_likes = self.data.iloc[idx]['likes']
        target = torch.tensor([log_likes], dtype=torch.float32)
        
        return inputs['pixel_values'].squeeze(0), target

# Model class (unchanged)d
class ViTRegressionModel(nn.Module):
    def __init__(self, pretrained_model_name='google/vit-base-patch16-224'):
        super(ViTRegressionModel, self).__init__()
        self.vit = ViTModel.from_pretrained(pretrained_model_name)
        self.regressor = nn.Sequential(
            nn.Linear(self.vit.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
    
    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output
        return self.regressor(pooled_output)

# Load and preprocess data
data = pd.read_csv('./instagram_data.csv')
data['image_path'] = data['image_path'].apply(lambda x: x.replace('../Data/insta_data/', './insta_data/'))

# !Uncomment to drop comments
# data = data.drop(columns=['log_no_of_comments'])

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
train_dataset = InstagramDataset(train_data, feature_extractor)
test_dataset = InstagramDataset(test_data, feature_extractor)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model = ViTRegressionModel()
# model.load_state_dict(torch.load('./core/trained_models/best_model_notransform.pth'))

# Freeze the ViT layers initially
for param in model.vit.parameters():
    param.requires_grad = False

# Use a higher learning rate for the regression head
optimizer = optim.AdamW(model.regressor.parameters(), lr=1e-3, weight_decay=0.01)

# Define the loss function (criterion)
criterion = nn.MSELoss()

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

num_epochs = 100
best_val_loss = float('inf')
patience = 3
no_improve = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    # Gradually unfreeze ViT layers
    if epoch == 10:  # After 10 epochs, unfreeze the last 2 layers of ViT
        for param in model.vit.encoder.layer[-2:].parameters():
            param.requires_grad = True
        optimizer.add_param_group({'params': model.vit.encoder.layer[-2:].parameters(), 'lr': 1e-5})
    
    for pixel_values, targets in train_loader:
        pixel_values, targets = pixel_values.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(pixel_values)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for pixel_values, targets in test_loader:
            pixel_values, targets = pixel_values.to(device), targets.to(device)
            outputs = model(pixel_values)
            val_loss += criterion(outputs, targets).item()
    
    val_loss /= len(test_loader)
    scheduler.step(val_loss)
    
    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {running_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}')
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), './core/trained_models/best_model_notransform_nocomments.pth')
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print("Early stopping")
            break

# Load best model for evaluation
model.load_state_dict(torch.load('./core/trained_models/best_model_notransform_nocomments.pth'))
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for pixel_values, targets in test_loader:
        pixel_values, targets = pixel_values.to(device), targets.to(device)
        outputs = model(pixel_values)
        y_true.extend(targets.cpu().numpy())
        y_pred.extend(outputs.cpu().numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f'Mean Squared Error: {mse:.4f}')
print(f'Mean Absolute Error: {mae:.4f}')
print(f'R-squared: {r2:.4f}')

plt.figure(figsize=(10, 10))
plt.scatter(y_true, y_pred, alpha=0.5)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
plt.xlabel('Actual Likes')
plt.ylabel('Predicted Likes')
plt.title('Actual vs Predicted Likes for ViT Model: Without Comments')
plt.legend(['Perfect Prediction', 'Model Predictions'])
plt.xscale('log')
plt.yscale('log')
plt.grid(True)
plt.show()