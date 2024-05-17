import glob

import torch
import numpy as np
import seaborn as sns
import torch.nn as nn
import streamlit as st
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from model import PneumoniaModel
from data_loader import PneumoniaDataset


def load_model(checkpoint: dict):
    """
    Load the model from a checkpoint.

    Args:
        checkpoint (dict): A dictionary containing the model state dict and configuration.

    Returns:
        model (nn.Module): The loaded PyTorch model.
    """
    cfg = OmegaConf.create(checkpoint["config"])
    model = PneumoniaModel(config=cfg.model)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str]):
    """
    Plot the confusion matrix using seaborn heatmap.

    Args:
        cm (ndarray): Confusion matrix.
        class_names (list): List of class names.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    st.pyplot(plt)


def evaluate_model(model: nn.Module, val_loader: DataLoader):
    """
    Evaluate the model on the validation set and compute the confusion matrix and accuracy.

    Args:
        model (nn.Module): The PyTorch model to evaluate.
        val_loader (DataLoader): DataLoader for the validation set.

    Returns:
        cm (ndarray): Confusion matrix.
        accuracy (float): Accuracy of the model on the validation set.
    """
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images, labels.float()
            outputs = model(images)
            predicted = outputs.round().squeeze().cpu().numpy()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted)

    cm = confusion_matrix(y_true, y_pred)
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    return cm, accuracy


def main():
    """
    Main function to run the Streamlit app.
    """
    st.title("Confusion Matrix for Pneumonia Detection Model Checkpoints")

    # List all checkpoint files
    checkpoint_files = [f for f in glob.glob("checkpoints/*.pth")]

    # Select a checkpoint file
    checkpoint_path = st.selectbox("Select Checkpoint", checkpoint_files)

    # Load the selected checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    config = OmegaConf.create(checkpoint["config"])

    # Define transformations
    transform_list = [
        transforms.Resize(tuple(config.transforms.resize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.transforms.normalize.mean, std=config.transforms.normalize.std),
    ]
    transform = transforms.Compose(transform_list)

    # Load the validation dataset
    val_dataset = PneumoniaDataset(root_dir="data/test", transform=transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Display configuration details
    config.run_name = None  # OmegaConf has problems with interpolation in this field
    config_serializable = OmegaConf.to_object(config)
    st.json(config_serializable, expanded=False)

    # Generate confusion matrix on button click
    if st.button("Generate Confusion Matrix"):
        if checkpoint_path:
            model = load_model(checkpoint)
            cm, accuracy = evaluate_model(model, val_loader)
            plot_confusion_matrix(cm, ["Normal", "Pneumonia"])
            st.write(f"Accuracy on test dataset: {accuracy}")


if __name__ == "__main__":
    main()
