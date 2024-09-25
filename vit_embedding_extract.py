import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from transformers import ViTModel, ViTFeatureExtractor

# Load the ViT model and feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTModel.from_pretrained('google/vit-base-patch16-224')

# Read the CSV file
data = pd.read_csv('./instagram_data.csv')
data['image_path'] = data['image_path'].apply(lambda x: x.replace('../Data/insta_data/', './insta_data/'))

# Function to extract embeddings from an image
def extract_image_embedding(image_path):
    image = Image.open(image_path)
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Loop through each row in the CSV and extract embeddings
embeddings = []
for idx, row in data.iterrows():
    image_path = row['image_path']
    embedding = extract_image_embedding(image_path)
    embeddings.append(embedding)

# Convert list of embeddings to a DataFrame and concatenate with the original CSV
embedding_df = pd.DataFrame(embeddings, columns=[f'img_emb_{i}' for i in range(embedding.shape[0])])
data_with_img_embeddings = pd.concat([data, embedding_df], axis=1)

# Save the new DataFrame to a CSV file
data_with_img_embeddings.to_csv('./data_with_img_embeddings.csv', index=False)
