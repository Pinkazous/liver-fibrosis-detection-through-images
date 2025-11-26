import torch
import torch.nn as nn
from torchvision import models

class TextureNetwork(nn.Module):
    def __init__(self, model_type='densenet', num_classes=5):
        super(TextureNetwork, self).__init__()
        self.model_type = model_type
        
        # Corrección del Warning: Usamos weights=None en lugar de pretrained=False
        # (Esto elimina el mensaje amarillo que te salía al inicio)
        if model_type == 'efficientnet':
            self.base = models.efficientnet_b0(weights=None)
            self.features = self.base.features
            out_channels = 1280 
        elif model_type == 'densenet':
            self.base = models.densenet121(weights=None)
            self.features = self.base.features
            out_channels = 1024
            
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        input_dim = out_channels * 2 
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(input_dim),
            nn.Dropout(p=0.5),
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        
        if self.model_type == 'densenet':
            # --- CORRECCIÓN CRÍTICA ---
            # Cambiamos inplace=True por inplace=False
            # Esto evita el RuntimeError con GradCAM
            x = nn.functional.relu(x, inplace=False)
            
        x_avg = self.avg_pool(x)
        x_max = self.max_pool(x)
        x = torch.cat((x_avg, x_max), dim=1) 
        x = self.classifier(x)
        return x