# CNN Image Caption Generator with LSTM

This project implements an **Image Caption Generator** using a combination of **Convolutional Neural Networks (CNN)** for image feature extraction and **Long Short-Term Memory (LSTM)** networks for generating natural language descriptions. The model is trained on the **Flickr30K dataset** to learn how to generate captions for images, leveraging the image features extracted by a pre-trained CNN (VGG16) and a sequence model (LSTM) for caption prediction.

## Table of Contents

- [Project Overview](#project-overview)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)
- [Results](#results)
- [References](#references)

## Project Overview

This project aims to automatically generate textual descriptions for images. The main steps include:

1. **Image Feature Extraction**: Using a pre-trained CNN (VGG16), the model extracts feature vectors from input images.
2. **Text Generation**: The LSTM is responsible for generating captions word by word, based on the features extracted from the CNN.
3. **Training**: The model is trained using a dataset of images with corresponding captions to learn the association between images and text.
4. **Caption Prediction**: After training, the model can generate a meaningful caption for a new image by predicting one word at a time until it generates a complete description.

## Code Architecture
![improved_image_captioning_architecture](https://github.com/user-attachments/assets/f3308ee6-38cd-453a-8422-a5f0cc45799f)



## Model Architecture

The model architecture is a combination of:
- **VGG16** for **image feature extraction**. The pre-trained VGG16 model is used to extract 4096-dimensional feature vectors from images by taking the output of the second fully connected layer (`fc2`).
- **LSTM network** for **text generation**. The LSTM processes the sequence of words to predict the next word in the sequence.
- The final output is a sequence of words forming a caption for the input image.

### Model Layers:
1. **Encoder (CNN)**: VGG16 model is used as the encoder to extract image features.
2. **Decoder (LSTM)**: The LSTM processes sequences of tokenized text (captions) and is trained to predict the next word in the sequence.
3. **Dense Layers**: Fully connected layers after the LSTM to produce word predictions based on vocabulary size.

## Dataset

The model is trained on the **Flickr30K** dataset, which contains 31,000 images with five captions per image.

- **Images**: 31,000 images in total.
- **Captions**: Each image has five human-annotated captions describing the content of the image.

## Requirements

To run this project, you will need the following libraries and frameworks:

- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- NLTK
- tqdm
- Pillow (PIL)
- Streamlit (for the interface)
- pickle (for saving and loading model/data)

You can install the required packages by running:

```bash
pip install -r requirements_cnn_Lstm.txt
```
## Installation

```bash
git clone git@github.com:spaarky/portfolio.git
```

## Usage
```bash
streamlit run interface.py
```
## Evaluate

To evaluate the model, you can use the BLEU score, which is a popular metric for evaluating text generation tasks like image captioning. The model is tested on a separate validation set, and the BLEU score is computed for different n-gram levels (e.g., BLEU-1, BLEU-2).

```bash
# BLEU score evaluation
bleu_score_1 = corpus_bleu(actual_captions, predicted_captions, weights=(1.0, 0, 0, 0))
bleu_score_2 = corpus_bleu(actual_captions, predicted_captions, weights=(0.5, 0.5, 0, 0))
```

## Results

### Tableau des Images et Captions

| Image | Caption |
|-------|---------|
| <img src="https://previews.123rf.com/images/tdietrich/tdietrich1309/tdietrich130900038/22119976-chien-qui-court-dans-un-pr%C3%A9.jpg" width="200"/> | brown dog is running on the grass |
| <img src="https://previews.123rf.com/images/oksun70/oksun701206/oksun70120600120/14105349-enfant-qui-joue-des-blocs-de-jouet-isol%C3%A9-sur-fond-blanc.jpg" width="200"/> | young girl in blue and orange stockings is playing with toy|
| <img src="https://img.freepik.com/photos-gratuite/homme-cuisine-homme-afro-americain-sejournant-dans-cuisine-homme-t-shirt-noir_1157-48443.jpg" width="200"/> | woman in black shirt and white shorts is cooking in kitchen |


## References

1. **LSTM Network**: For more information on LSTM networks, see [GeeksforGeeks](#LSTM-Network).
2. **VGG16 Model**: Details on the VGG16 model can be found in the [Keras documentation](#VGG16-Model).
3. **Flickr30k**: The dataset used is available on [Kaggle](#Flickr30k).
