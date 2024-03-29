import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from utils.data_utils.dataset_utils import StanfordCars, show_images, tensor_to_PIL
from utils.network_utils.network_utils import UNet

# torch.autograd.set_detect_anomaly(True)

def show_forward_process(image, diffusion_model, timesteps=200, num_images=10):
    stepsize = int(timesteps / num_images)

    plt.figure(figsize=(12, 12))
    for i in range(0, timesteps, stepsize):
        time = torch.Tensor([i]).type(torch.int64)
        image_nosiy, noise = diffusion_model.forward_diffusion(image.unsqueeze(0), time)
        image_pil = tensor_to_PIL(image_nosiy.squeeze())
        plt.subplot(1, num_images + 1, i // stepsize + 1)
        plt.imshow(image_pil)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

class DiffusionModel:
    def __init__(self, unet: UNet, 
                 beta_start=0.0001, beta_end=0.02, timesteps=200, 
                 lr=1e-3, epochs=200, device='cpu'):
        self.device = device
        self.timesteps = timesteps
        self.setup_noise_schedule(beta_start, beta_end, timesteps)

        self.network = unet.to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.epochs = epochs

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
        
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[timestep][:, None, None, None]
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[timestep][:, None, None, None]
        
        noisy_image = sqrt_alpha_cumprod * image + sqrt_one_minus_alpha_cumprod * noise
        
        return noisy_image, noise

    def train(self, dataloader):
        for epoch in range(self.epochs):
            for step, batch in enumerate(dataloader):
                batch = batch.to(self.device)
                self.optimizer.zero_grad()

                time = torch.randint(0, self.timesteps, (batch.size(0), ), device=self.device, dtype=torch.int64)
                image_noisy, noise = diffusion_model.forward_diffusion(batch, time)
                noise_pred = diffusion_model.network(image_noisy, time)
                loss = F.l1_loss(noise, noise_pred)

                loss.backward()
                self.optimizer.step()

                if epoch % 5 == 0 and step == 0:
                    print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()}")
                    self.sample_plot_image()

    @torch.no_grad()
    def sample_timestep(self, image, time):
        pass
    
    @torch.no_grad()
    def sample_plot_image(self):
        pass


if __name__ == '__main__':
    ##### 1. Load dataset #####
    dataset = StanfordCars()
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)
    # show_images(dataset=dataset, num_samples=8)

    ##### 2. Train the diffusion model(Forward process) #####
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    timesteps = 300

    unet = UNet()
    diffusion_model = DiffusionModel(unet=unet, device=device, timesteps=timesteps)
    # show_forward_process(image=next(iter(dataloader))[0], diffusion_model=diffusion_model, timesteps=timesteps)
    diffusion_model.train(dataloader=dataloader)

    ##### 3. Evaluate the diffusion model(Reverse process) #####
