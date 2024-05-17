import os

from PIL import Image
from omegaconf import DictConfig
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset


class PneumoniaDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        self.labels = []

        for label, subdir in enumerate(["NORMAL", "PNEUMONIA"]):
            subdir_path = os.path.join(root_dir, subdir)
            for file_name in os.listdir(subdir_path):
                if file_name.endswith((".jpeg", ".jpg", ".png")):
                    self.image_files.append(os.path.join(subdir_path, file_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(img_name).convert("L")  # Convert to grayscale
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_data_loaders(batch_size, train_dir, val_dirs, transform_config: DictConfig):
    transform_list = [
        transforms.Resize(tuple(transform_config.resize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=transform_config.normalize.mean, std=transform_config.normalize.std),
    ]
    transform = transforms.Compose(transform_list)

    train_dataset = PneumoniaDataset(root_dir=train_dir, transform=transform)
    val_test_datasets = [PneumoniaDataset(root_dir=dir_path, transform=transform) for dir_path in val_dirs]

    # Combine validation and test datasets
    val_dataset = ConcatDataset(val_test_datasets)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
