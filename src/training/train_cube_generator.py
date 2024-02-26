import pandas as pd
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences


from ast import literal_eval
from prompt_to_token import tokenize_prompt
from transformers import GPT2Tokenizer


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


# Load data
path_to_csv = os.path.join(os.getcwd(), "src", "training", "data", "cubes", "csv", "meta_data.csv")

df = pd.read_csv(path_to_csv)

# Preprocess token sequences
df['tokens'] = df['tokens'].apply(lambda x: literal_eval(x))  # Convert stringified lists back to actual lists

# Assuming the maximum length of token sequences is known or can be calculated
max_sequence_length = max(df['tokens'].apply(len))
print(max_sequence_length)

# Pad token sequences to have the same length
token_sequences = pad_sequences(df['tokens'], maxlen=max_sequence_length, padding='post')

# Prepare dimensions data (targets)
dimensions_data = df[['width', 'height', 'depth']].values  # Convert to NumPy array

# normalize the dimensions data
dimensions_data = dimensions_data / 100  # Normalize the dimensions to be in the range [0, 1]


# Model parameters
vocab_size = 50257  # Assuming using GPT-2 tokenizer; adjust based on your tokenizer
embedding_dim = 64  # Can be tuned

# Define the model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length))
model.add(LSTM(units=64, return_sequences=True))
model.add(Dropout(0.5))  # Add dropout
model.add(LSTM(units=32))
model.add(Dense(units=3))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')  # MSE is commonly used for regression tasks

# Train the model
model.fit(token_sequences, dimensions_data, epochs=10, batch_size=32, validation_split=0.2)

# Save the model if needed

# Now the model is trained and can be used to predict dimensions for new prompts
# For example:
new_prompt = "Assemble a cube with the following sizes: 97cm, 45cm, and 3cm."
new_token_ids = tokenize_prompt(new_prompt, tokenizer)
new_token_sequence = pad_sequences([new_token_ids], maxlen=max_sequence_length, padding='post')
predicted_dimensions = model.predict(new_token_sequence)



print(f"Predicted dimensions for the prompt '{new_prompt}': {predicted_dimensions[0]*100}")


# save path to model directory
save_path = os.path.join(os.getcwd(), "src", "testing", "models")

model.save(f'{save_path}/cube_predict.h5')
