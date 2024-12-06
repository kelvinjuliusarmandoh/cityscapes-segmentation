
"""Script for auxiliary functions"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

# Save model
def save_model(model: torch.nn.Module,
               destination_folder: str,
               filename: str):
    """Save model to destination folder with filename format .pt or .pth"""

    weights_model = model.state_dict()
    save_to_path = os.path.join(destination_folder, filename)
    return torch.save(weights_model, save_to_path)

def load_model(model: torch.nn.Module,
               model_saved_path: str):
    """Load the model"""
    weights_model = torch.load(model_saved_path, weights_only=True)
    model.load_state_dict(weights_model)
    return model
    
def plot_result_history(result_history: Dict, EPOCHS: int):
    """Plotting loss and dice performance of model

    Args:
    --------
    results_history: Dictionaries of loss and dice
    EPOCHS: Iteration while training model

    Return:
    --------
    Figures about loss and dice 
    """
    epochs = range(EPOCHS)
    fig, axs = plt.subplots(1, 2, figsize=(8, 6))

    axs[0].plot(epochs, result_history["train_loss"], label="Training Loss")
    axs[0].plot(epochs, result_history["val_loss"], label="Validating Loss")
    axs[0].set_title("Loss Performance")
    axs[0].axis(False)
    axs[0].legend()

    axs[1].plot(epochs, result_history["train_dice"], label="Training dice")
    axs[1].plot(epochs, result_history["val_dice"], label="Validating dice")
    axs[1].set_title("Dice Performance")
    axs[1].axis(False)
    axs[1].legend()

    plt.imshow()
