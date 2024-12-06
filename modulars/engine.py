
"""Script for training and validation the data, script file called engine.py"""

import torch
from tqdm.auto import tqdm
import torch.nn.functional as F
from timeit import default_timer as timer

"""
Training Loop and Validation Loop Functions for training the model
"""

def train_loop(model: torch.nn.Module,
               dataloaders: torch.utils.data.DataLoader,
               loss_fn,
               dice_fn,
               optimizer: torch.optim.Optimizer,
               device: torch.device):
    """
    Iterating the data for each batch and train the model.

    Args:
    ---------

    model: Model for being trained
    dataloader: Data for the model while training
    loss_fn: Loss function for comparing ground truth label and
             prediction
    optimizer: Function for updating the parameter
    dice_fn: Dice Coeff metrics
    device: Device for training

    Return:
    ---------
    Training the model for every data in each batch.
    """
    train_loss, train_dice = 0, 0 

    scaler = torch.amp.GradScaler()
    model.to(device)
    model.train()
    for batch, (image, true_mask) in enumerate(dataloaders):
        image, true_mask = image.to(device).float(), true_mask.to(device).long()
        true_mask = true_mask.squeeze() # (Batch, H, W)

        with torch.autocast(device_type=device, dtype=torch.float16):
            mask_pred_logits = model(image) # (Batch, N_classes, H, W)
            mask_pred_probs = torch.softmax(mask_pred_logits, dim=1)
            mask_pred_labels = torch.argmax(mask_pred_probs, dim=1)

            mask_pred_ohe = F.one_hot(mask_pred_labels, num_classes=20).float()
            true_mask_ohe = F.one_hot(true_mask, num_classes=20).float()
            
            loss = loss_fn(mask_pred_logits, true_mask)
            dice = dice_fn(mask_pred_ohe, true_mask_ohe)
    
        train_loss += loss
        train_dice += dice

        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    train_loss /= len(dataloaders)
    train_dice /= len(dataloaders)

    return train_loss, train_dice

def val_loop(model: torch.nn.Module,
             dataloaders: torch.utils.data.DataLoader,
             loss_fn,
             dice_fn,
             device: torch.device):
    """
    Iterating data in every batch and evaluating the model.
    
    Args:
    ---------
    model: Model for being evaluated
    dataloader: Data for the model while training
    loss_fn: Loss function for comparing ground truth label and
             prediction
    dice_fn: Dice Coeff metrics
    device: Device for training

    Return:
    ----------
    Evaluating the model for every batches.
    """

    val_loss, val_dice = 0, 0
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for batch, (image, true_mask) in enumerate(dataloaders):
            image, true_mask = image.to(device).float(), true_mask.to(device).long()
            true_mask = true_mask.squeeze() # (Batch, Height, Width)

            with torch.autocast(device_type=device, dtype=torch.float16):
                mask_pred_logits = model(image) # (Batch, N_Classes, H, W)
                mask_pred_probs = torch.softmax(mask_pred_logits, dim=1)
                mask_pred_labels = torch.argmax(mask_pred_probs, dim=1)

                mask_pred_ohe = F.one_hot(mask_pred_labels, num_classes=20).float()
                true_mask_ohe = F.one_hot(true_mask, num_classes=20).float()
                
                loss = loss_fn(mask_pred_logits, true_mask)
                dice = dice_fn(mask_pred_ohe, true_mask_ohe)
            
            val_loss += loss
            val_dice += dice
            
    val_loss = val_loss / len(dataloaders)
    val_dice = val_dice / len(dataloaders)
    return val_loss, val_dice

def train_model(model: torch.nn.Module,
                train_dataloaders: torch.utils.data.DataLoader,
                val_dataloaders: torch.utils.data.DataLoader,
                loss_fn,
                dice_fn,
                optimizer: torch.optim.Optimizer,
                scheduler,
                device=torch.device,
                EPOCHS: int=5):
    """
    Train the model for every epochs.

    Args:
    --------
    model: Model for being trained for every epochs
    train_dataloaders: Dataloaders for training,
    val_dataloaders: Dataloaders for validating,
    loss_fn: Loss function,
    dice_fn: Dice coefficient,
    device: Device for training and validating model,
    EPOCHS: Iteration for training and validating model

    Return:
    ---------
    Iterating every epochs and training the model. Returning 
    loss and dice for training and validating data.
    """

    results = {
        "train_loss": [],
        "train_dice": [],
        "val_loss": [],
        "val_dice": []
    }

    start_timer = timer()
    
    for epoch in tqdm(range(EPOCHS)):
        print(f"Epochs: {epoch + 1}\n----------")
        # Training loop
        train_loss, train_dice = train_loop(model,
                                            dataloaders=train_dataloaders,
                                            loss_fn=loss_fn,
                                            dice_fn=dice_fn,
                                            optimizer=optimizer,
                                            device=device)

        # Validating loop
        val_loss, val_dice = val_loop(model,
                                      dataloaders=val_dataloaders,
                                      loss_fn=loss_fn,
                                      dice_fn=dice_fn,
                                      device=device)
        
        # Scheduler
        scheduler.step(val_loss)

        print(f"Train loss: {train_loss:.3f} | Train dice: {train_dice:.3f} | Val loss: {val_loss:.3f} | Val dice: {val_dice:.3f}\n")
        
        results["train_loss"].append(train_loss.item())
        results["train_dice"].append(train_dice.item())
        results["val_loss"].append(val_loss.item())
        results["val_dice"].append(val_dice.item())

    end_timer = timer()
    training_process_time = end_timer - start_timer
    print(f"Training process takes time: {training_process_time:.3f} seconds")

    return results
