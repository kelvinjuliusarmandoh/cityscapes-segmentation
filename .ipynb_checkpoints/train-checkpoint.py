"""Main Script for training model, called train.py"""

import os
import torch
import torch.nn as nn
from modulars import data_setup, engine, model, utils
from torch.utils.data import DataLoader
from torchmetrics.segmentation import GeneralizedDiceScore
import albumentations as A
import argparse

# Define parser
parser = argparse.ArgumentParser(description="Hyperaparameters Setting")

# Create an argument for number of epochs
parser.add_argument("--num_epochs",
                    type=int,
                    default=10,
                    help="Number of epoch to train for")

# Create an argument for number of batch size
parser.add_argument("batch_size",
                    type=int,
                    default=4,
                    help="Number of images for each batch")

# Create an argument for learning rate
parser.add_argument("learning_rate",
                    type=float,
                    default=1e-3,
                    help="Learning rate for training the model")

# Create an argument for number of classes of segmentation 
parser.add_argument("num_classes",
                    type=int,
                    default=20,
                    help="Number of segmentation")

# Get our arguments from parser
args = parser.parse_args()

# Setting up Hyperparameters
NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
N_CLASSES = args.num_classes

# Directory path and Directory data list
TRAIN_DIR_PATH = os.path.join(os.getcwd(), "data", "cityscapes_data", "train")
TRAIN_DIR_ARRAY = os.listdir(TRAIN_DIR_PATH)
VAL_DIR_PATH = os.path.join(os.getcwd(), "data", "cityscapes_data", "val")
VAL_DIR_ARRAY = os.listdir(VAL_DIR_PATH)
        
# Setting target device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Transformers for training and validating data
train_transformer = A.Compose([
    A.Resize(128, 128, interpolation=1),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
val_transformer = A.Compose([
    A.Resize(128, 128),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Color label mapping
color_label_mapping = data_setup.color_label_mapping

# Dataset
train_ds = data_setup.CityScapesDatasets(dir_path=TRAIN_DIR_PATH,
                                         dir_array=TRAIN_DIR_ARRAY,
                                         process_mask_func=data_setup.process_mask,
                                         transform=train_transformer)
val_ds = data_setup.CityScapesDatasets(dir_path=VAL_DIR_PATH,
                                       dir_array=VAL_DIR_ARRAY,
                                       process_mask_func=data_setup.process_mask,
                                       transform=val_transformer)

# DataLoader
train_dl = DataLoader(train_ds,
                      batch_size=BATCH_SIZE,
                      shuffle=True,
                      num_workers=0)
val_dl = DataLoader(val_ds,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=0)
batch = next(iter(train_dl))

# Model
eff_segnet_model = model.EfficientSegNet(n_classes=N_CLASSES)

# Loss, Dice, Optimizer, and Scheduler
loss_fn = nn.CrossEntropyLoss().to(device)
dice_fn = GeneralizedDiceScore(num_classes=N_CLASSES).to(device)
optimizer = torch.optim.AdamW(params=eff_segnet_model.parameters(),
                              lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                       mode='min',
                                                       patience=10)

# Train the model
effsegnet_result = engine.train_model(model=eff_segnet_model,
                                      train_dataloaders=train_dl,
                                      val_dataloaders=val_dl,
                                      loss_fn=loss_fn,
                                      dice_fn=dice_fn,
                                      optimizer=optimizer,
                                      scheduler=scheduler,
                                      device=device,
                                      EPOCHS=NUM_EPOCHS)

# Plot result history
utils.plot_result_history(result_history=effsegnet_result,
                          EPOCHS=NUM_EPOCHS)

# Save the model
utils.save_model(model=eff_segnet_model,
                 destination_folder=os.getcwd(),
                 filename="eff_segnet_model.pth")
