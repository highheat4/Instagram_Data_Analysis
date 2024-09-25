import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from transformers import ViTModel, ViTFeatureExtractor
from sklearn.decomposition import PCA

# Load the ViT model and feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTModel.from_pretrained('google/vit-base-patch16-224')

# Read the CSV file
data = pd.read_csv('./new_instagram_data.csv')
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
    try:
        embedding = extract_image_embedding(image_path)
        embeddings.append(embedding)
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        continue  # Skip this image if there's an error

# Convert list of embeddings to a DataFrame
embedding_df = pd.DataFrame(embeddings, columns=[f'img_emb_{i}' for i in range(embedding.shape[0])])
data_with_img_embeddings = pd.concat([data, embedding_df], axis=1)

# Perform PCA for different numbers of components and save to separate CSV files
pca_components = [5, 20, 50, 100, 500]
for n_components in pca_components:
    pca = PCA(n_components=n_components)
    pca_embeddings = pca.fit_transform(embedding_df)  # Apply PCA on embeddings
    pca_embedding_df = pd.DataFrame(pca_embeddings, columns=[f'img_pca_emb_{i}' for i in range(n_components)])
    
    # Combine with original data (if needed) or save separately
    data_with_pca_embeddings = pd.concat([data.reset_index(drop=True), pca_embedding_df], axis=1)
    
    # Save to CSV
    pca_filename = f'./core/new_csvs/data_with_img_pca_embeddings_{n_components}.csv'
    data_with_pca_embeddings.to_csv(pca_filename, index=False)
    print(f'Saved PCA embeddings with {n_components} dimensions to {pca_filename}')

# Save the DataFrame with image embeddings to a CSV file
data_with_img_embeddings.to_csv('./core/new_csvs/data_with_img_embeddings.csv', index=False)
