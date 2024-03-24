import os
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset

def show_images(dataset, num_samples, cols=4):
    """ 
        Plots some samples from dataset 
    """
    plt.figure(figsize=(12, 12))
    for i, img in enumerate(dataset):
        if i == num_samples:
            break
        plt.subplot(int(num_samples/cols) + 1, cols, i + 1)
        plt.imshow(img)
    plt.show()

class StanfordCars(Dataset):
    """
        Load StanfordCars Dataset locally.
    """
    def __init__(self, root_path='./data/stanford_cars/archive/cars_train/cars_train/', transform=None):
        self.root_path = root_path
        self.image_paths = [os.path.join(root_path, file) for file in os.listdir(root_path)]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image
    
if __name__ == '__main__':
    dataset = StanfordCars()
    show_images(dataset, num_samples=8)