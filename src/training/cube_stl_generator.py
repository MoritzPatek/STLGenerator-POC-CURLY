"""
This file contains the code to generate the STL files for the cubes used for training the model.

Author: Moritz Patek 2024
"""

import hashlib
from stl import mesh
import numpy as np

from prompts import prompts
import random
import argparse
import os

# Function to create a cube at the origin
def create_cube(width, height, depth):
    # Define the 8 vertices of the cube
    vertices = np.array([
        [0, 0, 0], [width, 0, 0], [width, height, 0], [0, height, 0],
        [0, 0, depth], [width, 0, depth], [width, height, depth], [0, height, depth]
    ])

    # Define the 12 triangles composing the cube
    faces = np.array([
        [0, 3, 1], [1, 3, 2],  # Front
        [0, 4, 7], [0, 7, 3],  # Left
        [4, 5, 6], [4, 6, 7],  # Back
        [5, 1, 2], [5, 2, 6],  # Right
        [2, 3, 6], [3, 7, 6],  # Top
        [0, 1, 5], [0, 5, 4]   # Bottom
    ])

    # Create the mesh
    cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            cube.vectors[i][j] = vertices[f[j], :]

    return cube


def get_dimensions():
    """
    This function returns random dimensions for the cube.

    Returns:
        width: int
        height: int
        depth: int
    """
    width = random.randint(1, 100)
    height = random.randint(1, 100)
    depth = random.randint(1, 100)

    return width, height, depth

def get_prompt(width, height, depth):
    """
    This function returns a random prompt with the dimensions filled in.

    Args:
        width: int
        height: int
        depth: int

    Returns:
        prompt: str
    """
    prompt = random.choice(prompts)
    prompt = prompt.format(x=width, y=height, z=depth)

    return prompt

def main(args):
    """
    This function generates the STL files for the cubes and saves the prompts and cube IDs in a CSV file.

    Args:
        args: argparse.Namespace
    
    Returns:
        None
    """

    number_of_cubes = args.number
    base_output_dir = os.path.join(os.getcwd(), "src", "training", "data", args.output)

    # Create the base output directory if it doesn't exist
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)

    # STL files directory
    output_dir_stl = os.path.join(base_output_dir, "stl")
    if not os.path.exists(output_dir_stl):
        os.makedirs(output_dir_stl)

    # CSV files directory
    output_dir_csv = os.path.join(base_output_dir, "csv")
    if not os.path.exists(output_dir_csv):
        os.makedirs(output_dir_csv)

    # Save the prompt and cube ID in the CSV
    csv_filename = os.path.join(output_dir_csv, "prompts.csv")
    with open(csv_filename, "a") as f:
        header = "cube_id,prompt\n"
        f.write(header)

    for i in range(number_of_cubes):
        width, height, depth = get_dimensions()
        prompt = get_prompt(width, height, depth)

        print(f"Generating cube for prompt: {prompt}")

        cube = create_cube(width, height, depth)

        # Generate cube ID based on the prompt and create the STL filename
        cube_id = hashlib.md5(prompt.encode()).hexdigest()
        stl_filename = os.path.join(output_dir_stl, f"cube_{cube_id}.stl")

        try:
            cube.save(stl_filename)
            print(f"Successfully saved: {stl_filename}")
        except Exception as e:
            print(f"Error saving {stl_filename}: {e}")

        # Save the prompt and cube ID in the CSV
        csv_filename = os.path.join(output_dir_csv, "prompts.csv")
        with open(csv_filename, "a") as f:
            # Save the cube ID and prompt in the CSV the promt does have commas we should use a different delimiter
            # because of the csv format

            f.write(f'"{cube_id}","{prompt}"\n')

if __name__ == "__main__":
    # set up ArgumentParser
    parser = argparse.ArgumentParser(description="Generate a cube STL file")
    parser.add_argument("--number", type=int, default=10, help="How many cubes to generate")
    parser.add_argument("--output", type=str, default="cubes", help="Where to save the STL files")
    args = parser.parse_args()

    main(args)