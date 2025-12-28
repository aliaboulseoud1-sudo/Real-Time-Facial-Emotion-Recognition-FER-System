import os
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch
import csv


class FERDatasetLoader:
    def __init__(
        self,
        train_dir: str,
        test_dir: str,
        batch_size: int = 32,
        image_size: tuple[int, int] = (224, 224),
        channels: int = 3,
        seed: int = 42,
        num_workers: int = 4,
        pin_memory: bool = True
    ):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.channels = channels
        self.seed = seed
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.Grayscale(num_output_channels=self.channels),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485] * self.channels,
                std=[0.229] * self.channels
            )
        ])

        self.base_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.Grayscale(num_output_channels=self.channels),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485] * self.channels,
                std=[0.229] * self.channels
            )
        ])

    def get_loaders(self):

        full_train_dataset = datasets.ImageFolder(
            root=self.train_dir,
            transform=self.train_transform
        )
        
        val_dataset_full = datasets.ImageFolder(
             root=self.train_dir,
             transform=self.base_transform     
        )

        total_samples = len(full_train_dataset)
        train_size = int(0.8 * total_samples)
        val_size   = total_samples - train_size


        generator = torch.Generator().manual_seed(self.seed)
        indices = torch.randperm(total_samples, generator=generator)
        train_indices = indices[:train_size]
        val_indices   = indices[train_size:]

        self.train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices)
        self.val_dataset   = torch.utils.data.Subset(val_dataset_full, val_indices)
        self.test_dataset = datasets.ImageFolder(
            root=self.test_dir,
            transform=self.base_transform
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

        return self.train_loader, self.val_loader, self.test_loader, total_samples

    def get_len(self):
        return {
            "train_samples": len(self.train_dataset),
            "val_samples": len(self.val_dataset),
            "test_samples": len(self.test_dataset),
            "train_batches": len(self.train_loader),
            "val_batches": len(self.val_loader),
            "test_batches": len(self.test_loader)
        }

    def get_classes(self):
        return self.train_dataset.dataset.classes

    @staticmethod
    def csv_to_images(csv_file: str, output_dir: str, image_size=(48, 48)):
        os.makedirs(output_dir, exist_ok=True)

        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                emotion = row['emotion']
                pixels = list(map(int, row['pixels'].split()))
                img_array = np.array(pixels, dtype=np.uint8).reshape(48, 48)

                img = Image.fromarray(img_array)
                img = img.resize(image_size)

                class_dir = os.path.join(output_dir, emotion)
                os.makedirs(class_dir, exist_ok=True)

                img.save(os.path.join(class_dir, f"{i}.png"))

        print(f"[DONE] CSV converted to image folders → {output_dir}")



if __name__ == "__main__":
    train_dir = r"fer2013\train"
    test_dir  = r"fer2013\test"

    dataset_loader = FERDatasetLoader(
        train_dir=train_dir,
        test_dir=test_dir,
        batch_size=32,
        image_size=(224, 224),
        channels=3
    )

    train_loader, val_loader, test_loader, total_samples = dataset_loader.get_loaders()
    lens = dataset_loader.get_len()

    print("Total Train Folder Samples:", total_samples)
    print("Train Samples :", lens["train_samples"])
    print("Val Samples   :", lens["val_samples"])
    print("Test Samples  :", lens["test_samples"])
    print("-" * 50)
    print("Train Batches :", lens["train_batches"])
    print("Val Batches   :", lens["val_batches"])
    print("Test Batches  :", lens["test_batches"])
    print("-" * 50)
    train_ratio = round(lens["train_samples"] / total_samples * 100, 2)
    val_ratio   = round(lens["val_samples"]   / total_samples * 100, 2)
    print("Train Ratio :", train_ratio, "%")
    print("Val Ratio   :", val_ratio, "%")
    print("-" * 50)
    print("Classes:", dataset_loader.get_classes())