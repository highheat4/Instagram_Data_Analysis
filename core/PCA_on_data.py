from sklearn.decomposition import PCA
import pandas as pd

# Load the CSV files with embeddings
data_with_img_embeddings = pd.read_csv('./data_with_img_embeddings.csv')
data_with_word_embeddings = pd.read_csv('./data_with_word_embeddings.csv')

# List of ablation levels
ablations = [5, 20, 50, 100, 500]

# Function to apply PCA and reduce dimensionality
def apply_pca(data, columns, ablation):
    pca = PCA(n_components=ablation)
    reduced_embeddings = pca.fit_transform(data[columns])
    reduced_df = pd.DataFrame(reduced_embeddings, columns=[f'pca_emb_{i}' for i in range(ablation)])
    return reduced_df

# Apply PCA on image embeddings
embedding_columns_img = [col for col in data_with_img_embeddings.columns if col.startswith('img_emb_')]
for ablation in ablations:
    pca_img = apply_pca(data_with_img_embeddings, embedding_columns_img, ablation)
    data_with_img_embeddings = pd.concat([data_with_img_embeddings, pca_img], axis=1)

# Apply PCA on word embeddings
embedding_columns_word = [col for col in data_with_word_embeddings.columns if col.startswith('word_emb_')]
for ablation in ablations:
    pca_word = apply_pca(data_with_word_embeddings, embedding_columns_word, ablation)
    data_with_word_embeddings = pd.concat([data_with_word_embeddings, pca_word], axis=1)

# Save the reduced datasets to new CSV files
data_with_img_embeddings.to_csv('./data_with_img_embeddings_pca.csv', index=False)
data_with_word_embeddings.to_csv('./data_with_word_embeddings_pca.csv', index=False)
