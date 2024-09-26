import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from transformers import ViTFeatureExtractor
from transformers import ViTModel, ViTFeatureExtractor
from sklearn.decomposition import PCA

# Define the same ViTRegressionModel as in vit_train.py
class ViTRegressionModel(nn.Module):
    def __init__(self, pretrained_model_name='google/vit-base-patch16-224'):
        super(ViTRegressionModel, self).__init__()
        self.vit = ViTModel.from_pretrained(pretrained_model_name)
        self.regressor = nn.Sequential(
            nn.Linear(self.vit.config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    
    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output  # CLS token embedding
        return pooled_output  # Return embeddings instead of regressor for extraction

# Load the trained ViT model
def load_trained_model(model_path):
    model = ViTRegressionModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Load the fine-tuned model
trained_model = load_trained_model('./core/trained_models/fine_tuned_vit_model.pth')

# Load feature extractor (same as during training)
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

# Function to extract embeddings from an image
def extract_image_embedding(image_path, feature_extractor, model):
    try:
        image = Image.open(image_path)
        image = image.convert('RGB')
        
        inputs = feature_extractor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            # Get CLS token embedding (pooled_output)
            pixel_values = inputs['pixel_values']  # Extract pixel values
            embedding = model(pixel_values).squeeze(0).cpu().numpy()  # Get pooled_output embedding
        
        return embedding
    
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Load your CSV data
data = pd.read_csv('./new_instagram_data.csv')
data['image_path'] = data['image_path'].apply(lambda x: x.replace('../Data/insta_data/', './insta_data/'))

# Loop through each row in the CSV and extract embeddings
embeddings = []
for idx, row in data.iterrows():
    image_path = row['image_path']
    embedding = extract_image_embedding(image_path, feature_extractor, trained_model)  # Pass both feature_extractor and model
    
    if embedding is not None:
        embeddings.append(embedding)
    else:
        print(f"Skipping image {image_path} due to an error.")

# Convert list of embeddings to a DataFrame
embedding_df = pd.DataFrame(embeddings, columns=[f'img_emb_{i}' for i in range(embeddings[0].shape[0])])
data_with_img_embeddings = pd.concat([data, embedding_df], axis=1)

# Perform PCA and save to CSV files
pca_components = [5, 20, 50, 100, 500]
for n_components in pca_components:
    pca = PCA(n_components=n_components)
    pca_embeddings = pca.fit_transform(embedding_df)
    pca_embedding_df = pd.DataFrame(pca_embeddings, columns=[f'img_pca_emb_{i}' for i in range(n_components)])
    
    # Combine with original data and save
    data_with_pca_embeddings = pd.concat([data.reset_index(drop=True), pca_embedding_df], axis=1)
    pca_filename = f'./core/new_csvs/data_with_img_pca_embeddings_{n_components}.csv'
    data_with_pca_embeddings.to_csv(pca_filename, index=False)
    print(f'Saved PCA embeddings with {n_components} dimensions to {pca_filename}')

# Save the DataFrame with image embeddings
data_with_img_embeddings.to_csv('./core/new_csvs/data_with_img_embeddings.csv', index=False)
