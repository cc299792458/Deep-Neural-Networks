from ..standard_auto_encoder import AutoEncoder




class VAE(AutoEncoder):
    def __init__(self, feature_size: int, config: dict = None, device: str = 'cpu', lr: float = 0.001, epochs: int = 20) -> None:
        super().__init__(feature_size, config, device, lr, epochs)
    
    def forward(self):
        pass

    