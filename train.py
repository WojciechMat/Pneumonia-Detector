import hydra
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from omegaconf import DictConfig
from torch.utils.data import DataLoader

import wandb
from model import PneumoniaModel
from data_loader import get_data_loaders


def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: DictConfig,
    epochs: int,
    learning_rate: float,
    device: torch.device,
):
    """
    Train the pneumonia detection model.

    Args:
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        config (DictConfig): Configuration dictionary.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        device (torch.device): Device to run the training on (CPU or GPU).
    """
    wandb.init(project="pneumonia-detection", name=config.run_name, config=config)
    model = PneumoniaModel(config.model).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    log_steps = config.logging.steps

    best_accuracy = 0.0
    checkpoint_path = hydra.utils.to_absolute_path(f"checkpoints/{config.run_name}.pth")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for step, (images, labels) in enumerate(tqdm(train_loader)):
            images, labels = images.to(device), labels.float().to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (step + 1) % log_steps == 0:
                avg_loss = running_loss / log_steps
                wandb.log({"epoch": epoch + 1, "step": step + 1, "train/loss": avg_loss})
                running_loss = 0.0
                val_accuracy = evaluate_model(model, val_loader, device)
                wandb.log({"epoch": epoch + 1, "validation/accuracy": val_accuracy})

                # Save checkpoint if the accuracy improves
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    checkpoint = {
                        "model": model.state_dict(),
                        "config": config,
                    }
                    torch.save(checkpoint, checkpoint_path)


def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
):
    """
    Evaluate the model on the validation dataset.

    Args:
        model (nn.Module): The trained PyTorch model.
        val_loader (DataLoader): DataLoader for the validation dataset.
        device (torch.device): Device to run the evaluation on (CPU or GPU).

    Returns:
        float: Accuracy of the model on the validation dataset.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.float().to(device)
            outputs = model(images)
            predicted = outputs.round().squeeze()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


@hydra.main(config_name="config", config_path="configs", version_base=None)
def main(config: DictConfig):
    """
    Main function to run the training script.

    Args:
        config (DictConfig): Configuration dictionary.
    """
    train_loader, val_loader = get_data_loaders(
        batch_size=config.training.batch_size,
        train_dir=config.training.train_dir,
        val_dirs=config.training.val_dirs,
        transform_config=config.transforms,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_model(
        train_loader,
        val_loader,
        config,
        epochs=config.training.epochs,
        learning_rate=config.training.learning_rate,
        device=device,
    )


if __name__ == "__main__":
    main()
