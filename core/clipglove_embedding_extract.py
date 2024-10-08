import torch
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
import gensim.downloader
from PIL import Image
import os
from sklearn.decomposition import PCA
import numpy as np

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# Load GloVe embeddings using gensim downloader
glove_vectors = gensim.downloader.load('glove-wiki-gigaword-300')

# Read the CSV with image embeddings
data = pd.read_csv('./new_instagram_data.csv')
data['image_path'] = data['image_path'].apply(lambda x: x.replace('../Data/insta_data/', './insta_data/'))

# Function to get the top-1 text label using CLIP
def get_top_label(image_path):
    try:
        # Construct the full image path
        full_image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'insta_data', image_path))
        image = Image.open(full_image_path)
        
        # Text labels to compare
        text_labels = [
            'woman', 'man', 'dog', 'cat', 'baby', 'sunset', 'beach', 'mountains', 'food', 'coffee',
            'selfie', 'couple', 'friends', 'family', 'wedding', 'birthday', 'graduation',
            'workout', 'yoga', 'travel', 'landmark', 'cityscape', 'nature', 'flowers',
            'fashion', 'makeup', 'hairstyle', 'celebrity', 'car', 'luxury',
            'restaurant', 'cocktail', 'wine', 'dessert', 'healthy food', 'puppy', 'kitten',
            'adventure', 'concert', 'festival', 'party', 'art', 'painting', 'sculpture',
            'home decor', 'DIY project', 'transformation', 'skincare', 'technology',
            'book', 'quote', 'meme', 'funny', 'prank', 'dance', 'music',
            'sports', 'gym', 'bicycle', 'motorcycle', 'surfing', 'skiing', 'hiking',
            'camping', 'road trip', 'airplane', 'hotel', 'pool', 'fireworks', 'rainbow',
            'storm', 'snow', 'autumn leaves', 'spring blossoms', 'Christmas', 'Halloween',
            'office', 'money', 'shopping', 'unboxing', 'jewelry', 'watch', 'sunglasses',
            'sneakers', 'red carpet', 'behind the scenes', 'podcast', 'drone shot',
            'underwater photo', 'black and white', 'vintage filter', 'street art', 'protest',
            'charity event', 'environmental', 'gardening', 'cooking', 'baking', 'smoothie',
            'latte art', 'craft beer', 'picnic', 'stargazing', 'full moon', 'meditation',
            'spa', 'new haircut', 'beard', 'fitness transformation', 'martial arts',
            'marathon', 'rock climbing', 'ballet', 'street performance', 'karaoke', 
            'vinyl record', 'gaming', 'cosplay', 'comic con', 'anime', 'action figure',
            'virtual reality', 'robot', 'electric car', 'solar panel', 'tiny house',
            'van life', 'farm animals', 'exotic pet', 'aquarium', 'safari', 'waterfall',
            'cave', 'desert', 'tropical island', 'sushi', 'barbecue', 'vegan food',
            'craft cocktail', 'espresso', 'bubble tea', 'bookstore', 'home office',
            'productivity', 'self-care', 'engagement ring', 'tattoo', 'piercing',
            'streetwear', 'vintage clothing', 'ball gown', 'leather jacket', 'athleisure',
            'swimsuit', 'hat', 'backpack', 'nail art', 'henna', 'theme park',
            'waterpark', 'escape room', 'glamping', 'cruise ship', 'scuba diving',
            'skydiving', 'hot air balloon', 'first class', 'passport', 'souvenir',
            'language learning', 'online course', 'podcast studio', 'work from home',
            'side hustle', 'cryptocurrency', 'thrift find', 'upcycling', 'zero waste',
            'minimalism', 'bucket list', 'new year resolution', 'before and after',
            'throwback', 'challenge', 'hack', 'slow motion', 'time lapse', 'ASMR',
            'mukbang', 'room tour', 'morning routine', 'night routine', 'what I eat in a day',
            'get ready with me', 'outfit of the day', 'flat lay', 'product placement', 'sponsorship',
            'giveaway', 'mental health', 'eco-friendly', 'sustainable fashion', 'climate change',
            'entrepreneur', 'motivation', 'AI art', 'NFT', 'crossfit', 'keto diet', 'home workout'
        ]
        
        # Process the image
        inputs = processor(text=text_labels, images=image, return_tensors="pt", padding=True)
        
        # Get image logits
        with torch.no_grad():
            logits_per_image = model(**inputs).logits_per_image
        
        # Get the top-1 index
        top1_index = logits_per_image[0].argmax().item()
        
        # Map index back to the text label
        top1_label = text_labels[top1_index]
        
        return top1_label
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return ""  # Return an empty string in case of an error

# Function to convert text labels into GloVe embeddings
def get_word_embedding(label):
    words = label.split()
    word_embeddings = [glove_vectors[word] for word in words if word in glove_vectors]
    
    if word_embeddings:
        # Average the word embeddings if the label has more than one word
        return np.mean(word_embeddings, axis=0)
    else:
        # Return a zero vector if the word is not found in GloVe
        return np.zeros(300)

# Loop through each row, get the top-1 label, and append word embeddings
word_embeddings = []
for idx, row in data.iterrows():
    image_path = row['image_path'].split('/')[-1]  # Get the filename only
    top_label = get_top_label(image_path)
    word_embedding = get_word_embedding(top_label)
    word_embeddings.append(word_embedding)

# Convert list of word embeddings to a DataFrame
word_embedding_df = pd.DataFrame(word_embeddings, columns=[f'word_emb_{i}' for i in range(300)])
data_with_word_embeddings = pd.concat([data, word_embedding_df], axis=1)

# Save the DataFrame with word embeddings to a CSV file
data_with_word_embeddings.to_csv('./core/new_csvs/data_with_word_embeddings.csv', index=False)

# Perform PCA for different numbers of components and save to separate CSV files
pca_components = [1, 2, 5, 20, 50, 100, 200]
for n_components in pca_components:
    pca = PCA(n_components=n_components)
    pca_embeddings = pca.fit_transform(word_embedding_df)  # Apply PCA on embeddings
    pca_embedding_df = pd.DataFrame(pca_embeddings, columns=[f'word_pca_emb_{i}' for i in range(n_components)])
    
    # Combine with original data (if needed) or save separately
    data_with_pca_embeddings = pd.concat([data.reset_index(drop=True), pca_embedding_df], axis=1)
    
    # Save to CSV
    pca_filename = f'./core/new_csvs/data_with_word_pca_embeddings_{n_components}.csv'
    data_with_pca_embeddings.to_csv(pca_filename, index=False)
    print(f'Saved PCA embeddings with {n_components} dimensions to {pca_filename}')
