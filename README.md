# Cube Dimension Predictor

## Overview
This project harnesses the power of a dynamic Recurrent Neural Network (RNN) to translate textual descriptions into 3D cube dimensions. Leveraging the GPT-2 tokenizer, it processes variable-length sequences of tokens derived from textual prompts and predicts the corresponding width, height, and depth of geometric cubes.

## Features
- **Dynamic RNN Architecture**: Utilizes LSTM layers to handle variable-length token sequences, allowing for flexible input sizes.
- **GPT-2 Tokenization**: Employs the GPT-2 tokenizer for efficient and effective tokenization of textual prompts.
- **Dimension Prediction**: Maps textual prompts to cube dimensions, providing a foundational step towards creative geometric modeling.

## Getting Started

### Prerequisites
- Python 3.x
- TensorFlow 2.x
- Transformers library
- Pandas
- Numpy

### Installation
1. Clone the repository:

```
git clone https://github.com/MoritzPatek/STLGenerator-POC-CURLY
```

2. Install required Python packages:

```
pip install tensorflow transformers pandas numpy
```

### Usage
1. Prepare your dataset in the format of `cube_id, prompt, width, height, depth, tokens` and save it as a CSV file.
2. Use the provided script to train the model with your dataset. Adjust the `path_to_csv` variable to point to your dataset file.
3. After training, use the model to predict cube dimensions by providing new textual prompts.

Example:
```
new_prompt = "Create a cube with dimensions of 10cm x 20cm x 30cm."
predicted_dimensions = predict_dimensions(new_prompt)
print(f"Predicted dimensions: {predicted_dimensions}")
```
