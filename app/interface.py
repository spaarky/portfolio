import os
import ssl
import pickle
import streamlit as st
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

# Disable SSL verification to avoid issues with loading resources
ssl._create_default_https_context = ssl._create_unverified_context

# Define directory paths for data and models
BASE_DIR = '/Users/grem/Documents/portfolio/app'
WORKING_DIR = '/Users/grem/Documents/portfolio/app/working'

# Load the pre-trained caption generation model
caption_model = load_model(os.path.join(WORKING_DIR, 'best_model.h5'))

# Load the pre-trained VGG16 model and extract features from the 'fc2' layer
base_model = VGG16(weights='imagenet')  # Load VGG16 model with ImageNet weights
vgg_model = Model(inputs=base_model.inputs, outputs=base_model.get_layer('fc2').output)  # Create a new model with 'fc2' layer as output

# Extract features from images in the specified directory
features = {}
directory = os.path.join(BASE_DIR, 'Images')  # Directory containing images

for img_name in tqdm(os.listdir(directory)):  # Iterate through all images in the directory
    # Load the image from file
    img_path = directory + '/' + img_name  # Path to the image file
    image = load_img(img_path, target_size=(224, 224))  # Load image and resize to 224x224
    # Convert image pixels to numpy array
    image = img_to_array(image)  # Convert the image to a numpy array
    # Reshape data for the VGG16 model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))  # Reshape image to (1, height, width, channels)
    # Preprocess image for VGG16
    image = preprocess_input(image)  # Preprocess image for VGG16 model
    # Extract features
    feature = vgg_model.predict(image, verbose=0)  # Predict features for the image
    # Get image ID
    image_id = img_name.split('.')[0]  # Extract image ID from filename (without extension)
    # Store feature in dictionary
    features[image_id] = feature  # Save extracted feature in the dictionary

# Save extracted features to a pickle file for later use
pickle.dump(features, open(os.path.join(WORKING_DIR, 'features.pkl'), 'wb'))

# Load the tokenizer object from a pickle file
with open(os.path.join(WORKING_DIR, 'features.pkl'), 'rb') as f:
    tokenizer = pickle.load(f)  # Load the tokenizer

# Load captions from a text file
with open(os.path.join(BASE_DIR, 'captions.txt'), 'r') as f:
    next(f)  # Skip header line if present
    captions_doc = f.read()  # Read the entire file contents

# Function to clean captions by removing unwanted characters and adding start/end tokens
def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            # Retrieve the caption and preprocess
            caption = captions[i].lower()  # Convert caption to lowercase
            caption = caption.replace('[^A-Za-z]', '')  # Remove special characters using regex
            caption = caption.replace('\s+', ' ')  # Remove extra spaces
            # Add start and end tokens, and filter out single-letter words
            caption = 'startseq ' + " ".join([word for word in caption.split() if len(word) > 1]) + ' endseq'
            captions[i] = caption  # Update the caption in the list

# Create a mapping of image IDs to their captions
mapping = {}
for line in captions_doc.split('\n'):  # Process each line of the captions file
    tokens = line.split(',')
    if len(line) < 2:
        continue  # Skip lines with insufficient data
    image_id, caption = tokens[0], tokens[1:]
    image_id = image_id.split('.')[0]  # Remove image extension
    caption = " ".join(caption)  # Join caption tokens into a single string
    if image_id not in mapping:
        mapping[image_id] = []  # Initialize list for new image ID
    mapping[image_id].append(caption)  # Append caption to the list for the image ID

# Clean captions to prepare them for tokenization
clean(mapping)

# Prepare all captions for tokenization
all_captions = []
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)  # Collect all captions in a list

# Tokenize the captions
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)  # Fit tokenizer on the list of captions
vocab_size = len(tokenizer.word_index) + 1  # Vocabulary size (including padding)
max_length = max(len(caption.split()) for caption in all_captions)  # Maximum length of captions

# Function to convert an index to a word using the tokenizer
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None  # Return None if index is not found

# Function to generate captions for a given image
def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'  # Start with the start token
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]  # Convert input text to sequence of integers
        sequence = pad_sequences([sequence], max_length)  # Pad sequence to ensure consistent length
        yhat = model.predict([image, sequence], verbose=0)  # Predict next word probabilities
        yhat = np.argmax(yhat)  # Get index of word with highest probability
        word = idx_to_word(yhat, tokenizer)  # Convert index to word
        if word is None:
            break  # Stop if no word is found
        in_text += " " + word  # Append predicted word to input text
        if word == 'endseq':
            break  # Stop if end token is generated

    # Remove start and end tokens from the generated caption
    final_caption = in_text.replace('startseq', '').replace('endseq', '').strip()
    return final_caption

# Function to preprocess and extract features from an image
def extract_features(image):
    image = img_to_array(image)  # Convert image to numpy array
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))  # Reshape for model input
    image = preprocess_input(image)  # Preprocess image for VGG16
    feature = vgg_model.predict(image, verbose=0)  # Extract features
    return feature

# Streamlit user interface
st.title("Image Caption Generator")

# Image upload widget
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the uploaded image
    image = load_img(uploaded_file, target_size=(224, 224))
    st.image(image, caption='Loaded Image.', use_column_width=True)

    # Button to generate caption
    if st.button('Generate Caption'):
        # Extract image features
        feature = extract_features(image)
        # Generate caption for the image
        caption = predict_caption(caption_model, feature, tokenizer, max_length)
        st.write("Generated Caption: ", caption)  # Display the generated caption
