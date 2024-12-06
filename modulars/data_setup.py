"""Script for setting up data, the file script called data_setup.py"""


"""
From mask above, our mask value has range 0-255. So, it should be in the class index range. So we need to change/convert it.
Reference : https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
"""

import numpy as np
import cv2
import torch
import os
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple

color_label_mapping = {
    (0, 0, 0): 0, # background
    (128, 64, 128): 1, #road
    (244, 35, 232): 2, # sidewalk
    (70, 70, 70): 3, # building
    (102, 102, 156): 4, # wall
    (190, 153, 153): 5, #fence
    (153, 153, 153): 6, # pole
    (250, 170, 30): 7, # traffic light
    (220, 220, 0): 8, # traffic sign
    (107, 142, 35): 9, # vegetation
    (152, 251, 152): 10, # terrain
    (70, 130, 180): 11, # sky
    (220, 20, 60): 12, # person
    (255, 0, 0): 13, # rider
    (0, 0, 142): 14, # car
    (0, 0, 70): 15, # truck
    (0, 60, 100): 16, # bus
    (0, 80, 100): 17, # train
    (0, 0, 230): 18, # motorcyle
    (119, 11,32): 19 # bicycle
}

def process_mask(color_label_mapping, rgb_mask):

    height, width, channel = rgb_mask.shape
    mask = np.zeros((height, width), dtype=np.uint8)

    # Convert mapping to numpy array 
    color_mapping = np.array(list(color_label_mapping.keys()))
    labels = np.array(list(color_label_mapping.values()))

    # Reshape RGB mask for pixel-wise distance computation
    reshaped_mask = rgb_mask.reshape(-1, 3) # Get index of each pixel. Example (128*128, 3)
    processed_mask = np.zeros(reshaped_mask.shape[0], dtype=np.uint8)
    
    # Compute distances and map to nearest color
    for i, pixel in enumerate(reshaped_mask):
        # print(color_mapping, pixel)
        distances = np.linalg.norm(color_mapping - pixel, axis=1)
        # print(f"Distances: {distances}")
        closest_color_idx = np.argmin(distances)
        # print(f"Closest color index: {closest_color_idx}")
        processed_mask[i] = labels[closest_color_idx]

    mask = processed_mask.reshape(height, width)
    
    # Log the processed mask output
    # print("Unique values in processed mask:", np.unique(mask))
    
    return mask
    
class CityScapesDatasets(Dataset):
    def __init__(self, 
                 dir_path:str, 
                 dir_array: List, 
                 process_mask_func=None, 
                 transform=None):
        self.dir_path = dir_path
        self.images_arr = [file for file in dir_array if file.endswith(".jpg") or file.endswith(".jpeg")]
        self.process_mask_func = process_mask_func
        self.transform = transform

    def __len__(self):
        return len(self.images_arr)

    def __getitem__(self, idx):
        image_mask_file = self.images_arr[idx]
        image_mask_path = os.path.join(self.dir_path, image_mask_file)
        image_mask_read = cv2.imread(image_mask_path, cv2.IMREAD_COLOR)
        image_mask_read = cv2.cvtColor(image_mask_read, cv2.COLOR_BGR2RGB)

        # Get image from image_mask combination
        image = image_mask_read[:, :256, :]
        
        # Get mask from image_mask combination
        mask = image_mask_read[:, 256:, :]
        
        # Set the transformer
        if self.transform:
            # print(image.shape, preprocess_mask.shape)
            transformed_image_mask = self.transform(image=image, mask=mask)
            image = transformed_image_mask['image']
            mask = transformed_image_mask['mask']


        # image transpose
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        image /= 255.
        
        # Preprocess mask RGB values -> class indexes value
        preprocess_mask = self.process_mask_func(color_label_mapping, mask)
        preprocess_mask = np.expand_dims(preprocess_mask, axis=-1)
        preprocess_mask = np.transpose(preprocess_mask, (2, 0, 1))

        # Change numpy array to torch 
        image = torch.from_numpy(image)
        preprocess_mask = torch.from_numpy(preprocess_mask)

        # One hot encoding true mask
        # one_hot_mask = F.one_hot(preprocess_mask.squeeze().long(), num_classes=20).permute(2, 0, 1)
        return image, preprocess_mask
