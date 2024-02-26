import os
import pandas as pd
from stl import mesh
from transformers import GPT2Tokenizer
from tqdm import tqdm  # Import the tqdm library

path_to_csv = os.path.join(os.getcwd(), "src", "training", "data", "cubes", "csv", "prompts.csv")
path_to_stl = os.path.join(os.getcwd(), "src", "training", "data", "cubes", "stl")

# Path to the new CSV file for the meta data
new_csv_file = os.path.join(os.getcwd(), "src", "training", "data", "cubes", "csv", "meta_data.csv")

# Read the csv file
df = pd.read_csv(path_to_csv, delimiter=',', quotechar='"')

# Load a pre-trained tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def get_bounding_box_dimensions(stl_file):
    model_mesh = mesh.Mesh.from_file(stl_file)
    min_x, max_x = min(model_mesh.x.flatten()), max(model_mesh.x.flatten())
    min_y, max_y = min(model_mesh.y.flatten()), max(model_mesh.y.flatten())
    min_z, max_z = min(model_mesh.z.flatten()), max(model_mesh.z.flatten())
    
    width = max_x - min_x
    height = max_y - min_y
    depth = max_z - min_z
    
    return width, height, depth

def tokenize_prompt(prompt, tokenizer):
    # Tokenize the prompt using the provided tokenizer
    tokens = tokenizer.encode(prompt, add_special_tokens=True)
    return tokens

meta_data = []

# Get the list of STL files
stl_files = [f for f in os.listdir(path_to_stl) if f.endswith(".stl")]

# Iterate through files in the STL directory with a progress bar
for file in tqdm(stl_files, desc="Processing STL files"):
    stl_file_path = os.path.join(path_to_stl, file)
    width, height, depth = get_bounding_box_dimensions(stl_file_path)

    # Extract the cube ID from the file name
    cube_id = file.replace("cube_", "").replace(".stl", "")

    # Find the prompt corresponding to the cube ID in the DataFrame
    prompt = df[df['cube_id'] == cube_id]['prompt'].iloc[0]

    # Tokenize the prompt
    token_ids = tokenize_prompt(prompt, tokenizer)

    # Collect metadata and tokens
    meta_data.append({
        "cube_id": cube_id,
        "prompt": prompt,
        "width": width,
        "height": height,
        "depth": depth,
        "tokens": token_ids
    })

# Convert meta_data list to DataFrame
meta_data_df = pd.DataFrame(meta_data)

# Save to new CSV file
meta_data_df.to_csv(new_csv_file, index=False)

print(f"Metadata saved to {new_csv_file}")
