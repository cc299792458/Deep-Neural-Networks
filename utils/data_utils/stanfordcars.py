import os
import math
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

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