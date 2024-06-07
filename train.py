import logging
import os
import json
from typing import Dict, Tuple
import random
import time

import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.models as models
from download import load_and_split_data

# Load configuration
with open("params.yaml", "r") as f:
    config = yaml.safe_load(f)


# Extract the directory path from the log file path
log_dir = os.path.dirname(config["logging"]["file"])

# Create the directory if it doesn't exist
os.makedirs(log_dir, exist_ok=True)

output_dir = os.path.dirname(config["artifacts"]["output_dir"])
os.makedirs(output_dir, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=config["logging"]["level"],
    format=config["logging"]["format"],
    handlers=[
        logging.FileHandler(config["logging"]["file"]),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Type hints
Dataset = Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset]
ModelOutput = Tuple[nn.Module, optim.Optimizer, nn.CrossEntropyLoss]

ind = config["training"]["optimizer"]["lr"]

wandb.init(
    # Set the project where this run will be logged
    project="lb5",
    # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
    name=f"run_{ind}",
    # Track hyperparameters and run metadata
    config={
    "learning_rate": config["training"]["optimizer"],
    "dataset": "PET",
    "epochs": config["training"]["epochs"],
    })

def create_model() -> ModelOutput:
    
    # Create your model
    model = models.resnet50(pretrained=config["model"]["pretrained"])
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, config["model"]["num_classes"])

    
    # Freeze base layers if specified
    if config["model"].get("freeze_base", False):
        for param in model.parameters():
            param.requires_grad = False

    # Create the optimizer
    optimizer_config = config["training"]["optimizer"]
    optimizer = getattr(optim, optimizer_config["name"])(
        model.parameters(), lr=optimizer_config["lr"]
    )

    # Create the loss function
    loss_fn = getattr(nn, config["training"]["loss"])()

    return model, optimizer, loss_fn


def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_fn: nn.CrossEntropyLoss,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, float]:
    model.to(device)
    best_test_accuracy = 0.0
    
    for epoch in range(config["training"]["epochs"]):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs.requires_grad = False

            for param in model.parameters():
                param.requires_grad = True

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        train_accuracy = 100.0 * train_correct / train_total
        val_accuracy = 100.0 * val_correct / val_total

        logger.info(f"Epoch {epoch+1}: Train Loss={train_loss/train_total:.4f}, Train Accuracy={train_accuracy:.2f}%")
        logger.info(f"Epoch {epoch+1}: Val Loss={val_loss/val_total:.4f}, Val Accuracy={val_accuracy:.2f}%")

        if val_accuracy > best_test_accuracy:
            if config["artifacts"]["save_best_model"]:
                torch.save(model.state_dict(), os.path.join(config["artifacts"]["output_dir"], "best_model.pth"))

        # Evaluate on test set after each epoch
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)

                test_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()

        test_accuracy = 100.0 * test_correct / test_total
        logger.info(f"Epoch {epoch+1}: Test Loss={test_loss/test_total:.4f}, Test Accuracy={test_accuracy:.2f}%")
        wandb.log({
                "train_loss": train_loss / train_total,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss / val_total,
                "val_accuracy": val_accuracy,
                "test_loss": test_loss / test_total,
                "test_accuracy": test_accuracy,
            },
            step=epoch,
        )
        


        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy

    # Save logs
    with open(os.path.join(config["artifacts"]["output_dir"], "training.log"), "w") as log_file:
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                log_file.write(handler.stream.getvalue())

def main() -> None:
    output_dir = config["artifacts"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess the dataset
    train_dataset, val_dataset, test_dataset = load_and_split_data(config["data"]["local_dir"])
       
        
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"])
    # Create the model, optimizer, and loss function
    model, optimizer, loss_fn = create_model()

    # Train the model
    train(model, optimizer, loss_fn, train_loader, val_loader, test_loader)

    wandb.finish()
        
        

if __name__ == "__main__":
    main()
