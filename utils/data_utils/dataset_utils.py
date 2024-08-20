import numpy as np
import matplotlib.pyplot as plt

from torchvision.utils import make_grid

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

if __name__ == '__main__':
    pass