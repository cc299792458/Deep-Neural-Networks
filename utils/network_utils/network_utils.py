import torch
import torch.nn as nn
import torch.optim as optim
import math

#NOTE: Not sure this form is absolutly the same to the pos encoding in "Attention is All You Need"
# Timesteps embedding
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super(SinusoidalPositionEmbeddings, self).__init__()
        
        self.dim = dim

    def forward(self, time):
        # Calculate the scale
        device = time.device
        half_dim = self.dim // 2
        scale = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * (-math.log(10000.0) / half_dim))
        
        # Calculate the time encodings
        time = time.unsqueeze(1)
        time_encodings = torch.zeros(time.size(0), self.dim, device=device)
        time_encodings[:, 0::2] = torch.sin(time * scale)
        time_encodings[:, 1::2] = torch.cos(time * scale)
        
        return time_encodings


class UNetBlock(nn.Module):
    def __init__(self, input_channels, output_channels, time_embedding_dim, is_upsample=False):
        super(UNetBlock, self).__init__()
        
        # Time-dependent transformation
        self.time_dense = nn.Linear(time_embedding_dim, output_channels)
        
        # Spatial transformation based on the operation mode (up or not)
        if is_upsample:
            self.initial_conv = nn.Conv2d(input_channels * 2, output_channels, kernel_size=3, padding=1)
            self.spatial_transform = nn.ConvTranspose2d(output_channels, output_channels, kernel_size=4, stride=2, padding=1)
        else:
            self.initial_conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
            self.spatial_transform = nn.Conv2d(output_channels, output_channels, kernel_size=4, stride=2, padding=1)
        
        self.final_conv = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        self.activation = nn.ReLU()
        self.batch_norm = nn.BatchNorm2d(output_channels)

    def forward(self, x, time_embedding):
        # Apply initial spatial transformation
        x = self.activation(self.batch_norm(self.initial_conv(x)))
        
        # Transform and integrate time information
        time_features = self.activation(self.time_dense(time_embedding)).unsqueeze(-1).unsqueeze(-1)
        x += time_features
        
        # Apply final spatial transformation and activate
        x = self.activation(self.batch_norm(self.final_conv(x)))
        
        # Upsample or downsample the feature map
        x = self.spatial_transform(x)
        
        return x


class UNet(nn.Module):
    def __init__(self, img_channels=3, img_size=64, out_channels=3):
        super(UNet, self).__init__()
        # Define image size, channels, and time embedding dimensions
        self.img_size = img_size
        time_embedding_dim = 32
        
        # Set the channel sizes for down and up sampling paths
        down_channels = [64, 128, 256, 512, 1024]
        up_channels = [1024, 512, 256, 128, 64]
        
        # Time embedding layer
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embedding_dim),
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.ReLU()
        )
        
        # Initial convolution layer to match the channel size of the first downsample layer
        self.initial_conv = nn.Conv2d(img_channels, down_channels[0], kernel_size=3, padding=1)
        
        # Downsample layers
        self.downsample_layers = nn.ModuleList([
            UNetBlock(input_channels=down_channels[i], output_channels=down_channels[i+1], time_embedding_dim=time_embedding_dim, is_upsample=False)
            for i in range(len(down_channels)-1)
        ])
        
        # Upsample layers
        self.upsample_layers = nn.ModuleList([
            UNetBlock(input_channels=up_channels[i], output_channels=up_channels[i+1], time_embedding_dim=time_embedding_dim, is_upsample=True)
            for i in range(len(up_channels)-1)
        ])
        
        # Final convolution layer to convert the feature map back to the target channel size
        self.final_conv = nn.Conv2d(up_channels[-1], out_channels, kernel_size=1)
    
    def forward(self, x, timestep):
        # Embed the timestep
        time_embed = self.time_embedding(timestep)
        
        # Process with initial convolution
        x = self.initial_conv(x)
        
        # Downsample, saving the output of each layer for the subsequent upsample connection
        skip_connections = []
        for layer in self.downsample_layers:
            x = layer(x, time_embed)
            skip_connections.append(x)
        
        # Upsample, using the saved downsample layer outputs for skip connections
        for layer, skip_connection in zip(self.upsample_layers, reversed(skip_connections)):
            x = torch.cat((x, skip_connection), dim=1)  # Concatenate for skip connection
            x = layer(x, time_embed)
        
        # Apply final convolution to produce the output
        x = self.final_conv(x)
        return x
    
if __name__ == '__main__':
    model = UNet()
    print(model)
    print("Num params:", sum(p.numel() for p in model.parameters()))

    