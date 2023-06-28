import os
from tqdm import tqdm
import numpy as np
import cv2

def divide_image(image, part_size):
    """
    crop image into n parts with given size
    image: opencv image (numpy ndarray: (h, w, c))
    part_size: tuple of (height, width)
    """
    height, width, _ = image.shape
    crops = []

    # Calculate the number of parts in each dimension
    n_parts_vertical = height // part_size[0]
    n_parts_horizontal = width // part_size[1]

    # Iterate over the image and extract each part
    for i in range(n_parts_vertical):
        for j in range(n_parts_horizontal):
            # Calculate the starting and ending indices for each part
            start_i = i * part_size[0]
            end_i = (i + 1) * part_size[0]
            start_j = j * part_size[1]
            end_j = (j + 1) * part_size[1]

            # Extract the part from the image
            part = image[start_i:end_i, start_j:end_j]
            crops.append(part)
    return crops

if __name__ == "__main__":
    root_dir = "./demo200/images"
    image_paths = []
    for filename in os.listdir(root_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', 'tif')):
            image_paths.append(os.path.join(root_dir, filename))

    crop_size = (1536, 1024)
    save_dir = f"{root_dir}_{crop_size[1]}"
    file_ext = "tiff"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for image_path in tqdm(image_paths):
        image = cv2.imread(image_path)
        crops = divide_image(image, crop_size)
        for idx, part in enumerate(crops):
            save_name = f"{os.path.basename(image_path).split('.')[0]}_part{idx}.{file_ext}"
            save_path = os.path.join(save_dir, save_name)
            cv2.imwrite(save_path, part)