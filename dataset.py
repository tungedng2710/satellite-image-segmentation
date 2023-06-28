import os
import numpy as np
import cv2
import torch
from utils import *

class MassachusettsDataset(torch.utils.data.Dataset):

    """Massachusetts Segmentation Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_rgb_values (list): RGB values of select classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            class_rgb_values=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        
        self.image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]
        self.mask_paths = [os.path.join(masks_dir, image_id) for image_id in sorted(os.listdir(masks_dir))]

        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read images and masks
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2RGB)
        
        # one-hot-encode the mask
        mask = one_hot_encode(mask, self.class_rgb_values).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        # return length of 
        return len(self.image_paths)


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
