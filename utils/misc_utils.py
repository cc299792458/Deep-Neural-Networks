import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torchvision.transforms.functional import to_pil_image

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_random_images_and_save(model, num_images, log_dir, image_size=28, latent_dim=2, z=None):
    model.eval()
    z = torch.randn(num_images, latent_dim).to(model.device) if z == None else z

    with torch.no_grad():
        generated_images = model.sample(z).cpu()
    
    cols = rows = math.ceil(math.sqrt(num_images))
    collage_width = cols * image_size
    collage_height = rows * image_size
    collage = Image.new('RGB', (collage_width, collage_height))

    for i, img_tensor in enumerate(generated_images):
        img = to_pil_image(img_tensor.view(-1, image_size, image_size))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        x_pos = (i % cols) * image_size
        y_pos = (i // cols) * image_size
        collage.paste(img, (x_pos, y_pos))
        if i == (rows * cols) - 1:
            break
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    collage_file_path = os.path.join(log_dir, "random_sample.png")
    collage.save(collage_file_path)
    print(f"Random sample saved to {collage_file_path}")

def generate_uniformly_distributed_images_and_save(model, num_images, z_ranges, log_dir, image_size=28, latent_dim=2):
    model.eval()
    
    ## Calculate z ##
    steps_per_dim = int(round(num_images ** (1. / latent_dim)))
    if latent_dim != len(z_ranges) or steps_per_dim ** latent_dim != num_images:
        raise ValueError("num_images must allow a perfect grid given latent_dim and z_ranges length.")
    meshgrid = [torch.linspace(start, end, steps=steps_per_dim) for start, end in z_ranges]
    grid = torch.stack(torch.meshgrid(*meshgrid), dim=-1).reshape(-1, latent_dim)
    z = grid.to(model.device)

    ## Sample images ##
    with torch.no_grad():
        generated_images = model.sample(z).cpu()
    
    ## Draw the picture ##
    cols, rows = steps_per_dim, steps_per_dim ** (latent_dim - 1) if latent_dim > 1 else 1
    collage_width = cols * image_size
    collage_height = rows * image_size
    collage = Image.new('RGB', (collage_width, collage_height))
    for i, img_tensor in enumerate(generated_images):
        img = to_pil_image(img_tensor.view(-1, image_size, image_size))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        x_pos = (i % cols) * image_size
        y_pos = (i // cols) * image_size
        collage.paste(img, (x_pos, y_pos))

    ## Save ##
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    collage_file_path = os.path.join(log_dir, "uniformly_distributed_sample.png")
    collage.save(collage_file_path)
    print(f"Uniformly distributed sample saved to {collage_file_path}")

def show_forward_process(image, diffusion_model, timesteps=200, num_images=10):
    stepsize = int(timesteps / num_images)
    plt.figure(figsize=(12, 12))
    for i in range(0, timesteps, stepsize):
        time = torch.Tensor([i]).type(torch.int64)
        image_nosiy, noise = diffusion_model.forward_diffusion(image.unsqueeze(0), time)
        image_pil = to_pil_image(image_nosiy.squeeze())
        plt.subplot(1, num_images + 1, i // stepsize + 1)
        plt.imshow(image_pil)
        plt.axis('off')
    plt.tight_layout()
    plt.show()