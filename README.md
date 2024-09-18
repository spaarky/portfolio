
# CNN-LSTM Image Caption Generator

This project is an image caption generator built using a combination of Convolutional Neural Networks (CNN) for image feature extraction and Long Short-Term Memory (LSTM) networks for caption generation. The model extracts features from images using a pre-trained VGG16 network and generates captions using an LSTM-based language model.

## Features

- **Image Feature Extraction**: Utilizes a pre-trained VGG16 CNN model to extract high-level features from input images.
- **Caption Generation**: Employs an LSTM-based language model to generate meaningful captions based on the extracted image features.
- **Streamlit Interface**: A user-friendly web interface built with Streamlit for uploading images and generating captions.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/cnn-lstm-caption-generator.git
    cd cnn-lstm-caption-generator
    ```

2. Install the required Python libraries:

    ```bash
    pip install -r requirements.txt
    ```

3. Download the pre-trained VGG16 weights (optional) or ensure TensorFlow handles model loading.

## Usage

### Training the Model

To train the captioning model:

```bash
python train.py
```

Make sure to have your image dataset and corresponding captions prepared in the correct format.

### Running the Streamlit App

To run the app and generate captions for your images:

```bash
streamlit run app.py
```

This will launch a web interface where you can upload images and get captions generated.

## Code Structure

- `app.py`: Streamlit web application for caption generation.
- `train.py`: Script to train the captioning model using image features and captions.
- `model.py`: Defines the CNN-LSTM model architecture.
- `data_generator.py`: Custom data generator to feed image-caption pairs to the model.
- `requirements.txt`: List of dependencies required for the project.

## Dataset

This model can be trained on any image-caption dataset like [Flickr8k](https://www.kaggle.com/adityajn105/flickr8k), [Flickr30k](https://www.kaggle.com/hsankesara/flickr-image-dataset), or MSCOCO dataset.

## References

1. [VGG16 - Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
2. [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555)
3. [Flickr8k Dataset](https://www.kaggle.com/adityajn105/flickr8k)
4. [Flickr30k Dataset](https://www.kaggle.com/hsankesara/flickr-image-dataset)
