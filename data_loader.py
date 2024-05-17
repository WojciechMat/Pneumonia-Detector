import os

from PIL import Image
from omegaconf import DictConfig
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset


class PneumoniaDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        """
        Initialize the PneumoniaDataset.

        Args:
            root_dir (str): Root directory containing subdirectories for each class (NORMAL and PNEUMONIA).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        self.labels = []

        # Iterate through the subdirectories (NORMAL and PNEUMONIA)
        for label, subdir in enumerate(["NORMAL", "PNEUMONIA"]):
            subdir_path = os.path.join(root_dir, subdir)
            for file_name in os.listdir(subdir_path):
                if file_name.endswith((".jpeg", ".jpg", ".png")):
                    self.image_files.append(os.path.join(subdir_path, file_name))
                    self.labels.append(label)

    def __len__(self):
        """
        Return the total number of samples.
        """
        return len(self.image_files)

    def __getitem__(self, idx: int):
        """
        Retrieve a sample and its corresponding label.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image, label) where image is the transformed image and label is the class label.
        """
        img_name = self.image_files[idx]
        image = Image.open(img_name).convert("L")  # Convert to grayscale
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_data_loaders(
    batch_size: int,
    train_dir: str,
    val_dirs: str,
    transform_config: DictConfig,
):
    """
    Create data loaders for training and validation datasets.

    Args:
        batch_size (int): Number of samples per batch.
        train_dir (str): Directory for training data.
        val_dirs (list of str): Directories for validation data.
        transform_config (DictConfig): Configuration for data transformations.

    Returns:
        tuple: (train_loader, val_loader) where train_loader is the DataLoader
        for the training set and val_loader is the DataLoader for the validation set.
    """
    # Define the transformations
    transform_list = [
        transforms.Resize(tuple(transform_config.resize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=transform_config.normalize.mean, std=transform_config.normalize.std),
    ]
    transform = transforms.Compose(transform_list)

    # Create the training dataset
    train_dataset = PneumoniaDataset(root_dir=train_dir, transform=transform)

    # Create the validation datasets
    val_test_datasets = [PneumoniaDataset(root_dir=dir_path, transform=transform) for dir_path in val_dirs]

    # Combine validation and test datasets
    val_dataset = ConcatDataset(val_test_datasets)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
