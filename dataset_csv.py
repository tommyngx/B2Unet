import os
import argparse
import random
import pandas as pd
from tqdm import tqdm

def create_dataset(input_folder):
    # Create lists to store information for the DataFrame
    ids = []
    image_names = []
    path_images = []
    path_masks = []
    splits = []

    # Get the list of images from the 'images' subfolder
    image_folder = os.path.join(input_folder, 'images')
    image_files = os.listdir(image_folder)

    # Iterate through each image file
    for image_file in tqdm(image_files, desc="Processing images"):
        # Check if the file is an image (you may want to add more image formats)
        if image_file.endswith(('.jpg', '.jpeg', '.png')):
            # Extract the ID and image name
            image_id, _ = os.path.splitext(image_file)
            # Create paths for images and masks
            image_path = os.path.join(image_folder, image_file)
            mask_path = os.path.join(input_folder, 'masks', f'{image_id}_mask.png')

            # Append information to the lists
            ids.append(image_id)
            image_names.append(image_file)
            path_images.append(image_path)
            path_masks.append(mask_path)

            # Randomly assign the split (80% train, 20% test)
            split = 'train' if random.uniform(0, 1) < 0.8 else 'test'
            splits.append(split)

    # Create a DataFrame
    df = pd.DataFrame({
        'ID': ids,
        'image_name': image_names,
        'path_images': path_images,
        'path_masks': path_masks,
        'split': splits
    })

    # Save the DataFrame to a CSV file
    output_path = os.path.join(input_folder, 'dataset.csv')
    df.to_csv(output_path, index=False)
    print(f'Dataset saved to {output_path}')

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Create dataset from images and masks.")
    parser.add_argument("input_folder", help="Path to the input folder containing 'images' and 'masks' subfolders.")
    args = parser.parse_args()

    # Check if the input folder exists
    if not os.path.exists(args.input_folder):
        print(f"Error: The specified input folder '{args.input_folder}' does not exist.")
    else:
        # Call the function to create the dataset
        create_dataset(args.input_folder)
