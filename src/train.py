import hydra
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from omegaconf import DictConfig

import wandb
from model import PneumoniaModel
from data_loader import get_data_loaders


def train_model(train_loader, val_loader, config, epochs, learning_rate, device):
    wandb.init(project="pneumonia-detection", config=config)
    model = PneumoniaModel(config.model).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    log_steps = config.logging.steps

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

    torch.save(model.state_dict(), "models/pneumonia_model.pth")


def evaluate_model(model, val_loader, device):
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
