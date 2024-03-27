import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset_utils import StanfordCars, show_images, tensor_to_PIL
from network_utils import UNet

def show_forward_process(image, diffusion_model, timesteps=200, num_images=10):
    stepsize = int(timesteps / num_images)

    plt.figure(figsize=(12, 12))
    for i in range(0, timesteps, stepsize):
        time = torch.Tensor([i]).type(torch.int64)
        processed_image, noise = diffusion_model.forward_diffusion(image, time)
        pil_image = tensor_to_PIL(processed_image)
        plt.subplot(1, num_images + 1, i // stepsize + 1)
        plt.imshow(pil_image)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

class DiffusionModel:
    def __init__(self, unet, beta_start=0.0001, beta_end=0.02, timesteps=200, device='cpu'):
        self.network = unet.to(device)
        self.device = device
        self.setup_noise_schedule(beta_start, beta_end, timesteps)

    def setup_noise_schedule(self, beta_start, beta_end, timesteps):
        betas = torch.linspace(beta_start, beta_end, steps=timesteps, device=self.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=self.device), alphas_cumprod[:-1]])

        self.betas = betas
        self.sqrt_alphas = torch.sqrt(alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_reciprocals_alphas = torch.sqrt(1.0 / alphas)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    def forward_diffusion(self, image, timestep):
        image = image.to(self.device)
        noise = torch.randn_like(image)
        
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[timestep]
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[timestep]
        
        noisy_image = sqrt_alpha_cumprod * image + sqrt_one_minus_alpha_cumprod * noise
        
        return noisy_image, noise


if __name__ == '__main__':
    ##### 1. Load dataset #####
    dataset = StanfordCars()
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)
    # show_images(dataset=dataset, num_samples=8)

    ##### 2. Train the diffusion model(Forward process) #####
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    timesteps = 200

    unet = UNet()
    diffusion_model = DiffusionModel(unet=unet, device=device, timesteps=timesteps)
    # show_forward_process(image=next(iter(dataloader))[0], diffusion_model=diffusion_model, timesteps=timesteps)

    ##### 3. Evaluate the diffusion model(Reverse process) #####