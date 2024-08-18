import os
import math
import glob
import random
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import make_grid


class StanfordCars(Dataset):
    """
    Load Stanford Cars Dataset from a local directory.
    """
    def __init__(self, img_size=64, 
                 root_path='./data/stanford_cars/archive/cars_train/cars_train/', 
                 transform=None):
        """
        Initializes the dataset.

        Parameters:
        - img_size: Integer, the size to which images are resized.
        - root_path: String, path to the directory containing images.
        - transform: torchvision.transforms, custom transformations for the images.
        """
        self.root_path = root_path
        self.image_paths = [os.path.join(root_path, file) for file in os.listdir(root_path)]
        self.transform = transform or transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Adjust to mean 0 and std 1
        ])

    def __len__(self):
        """Returns the number of images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, index):
        """Returns the image at the specified index after applying transformations."""
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image) if self.transform else image

def tensor_to_PIL(tensor):
    """
    Converts a tensor to a PIL Image.

    Parameters:
    - tensor: Torch tensor, the tensor to convert.
    
    Returns:
    - A PIL Image converted from the input tensor.
    """
    return transforms.ToPILImage()(tensor * 0.5 + 0.5)  # Un-normalize

def show_images(dataset, num_samples=8, cols=4):
    """
    Displays a grid of images from the dataset.

    Parameters:
    - dataset: StanfordCars, the dataset from which to show images.
    - num_samples: Integer, the number of samples to display.
    - cols: Integer, the number of columns in the grid.
    """
    plt.figure(figsize=(12, 12))
    for i in range(num_samples):
        img_tensor = dataset[i]
        img = tensor_to_PIL(img_tensor)
        plt.subplot(math.ceil(num_samples / cols), cols, i + 1)
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_data_from_dataloader(dataloader, device, type=None):
    if type == None:
        batch = next(iter(dataloader))
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(np.transpose(make_grid(batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
        plt.show()
    elif type == 'style_transfer':
        batch = next(iter(dataloader))
    
        images_A = batch['A'].to(device)
        images_B = batch['B'].to(device)

        plt.figure(figsize=(16, 8))
        plt.axis("off")
        plt.title("Training Images - Style A | Style B")
        
        plt.subplot(1, 2, 1)
        plt.imshow(np.transpose(make_grid(images_A, padding=2, normalize=True).cpu(), (1, 2, 0)))
        plt.title("Style A")
        
        plt.subplot(1, 2, 2)
        plt.imshow(np.transpose(make_grid(images_B, padding=2, normalize=True).cpu(), (1, 2, 0)))
        plt.title("Style B")
        
        plt.show()

def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image

class StyleTransferDataset(Dataset):
    def __init__(self, root, transform=None, unaligned=False, mode="train"):
        self.transform = transform
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, "%sA" % mode) + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "%sB" % mode) + "/*.*"))

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])

        if self.unaligned:
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)])

        # Convert grayscale images to rgb
        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = to_rgb(image_B)

        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

if __name__ == '__main__':
    dataset = StanfordCars()
    show_images(dataset, num_samples=16)
