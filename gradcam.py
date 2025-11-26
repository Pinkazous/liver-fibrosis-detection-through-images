import torch
import torch.nn.functional as F
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.feature_maps = None
        self.gradients = None
        self.handlers = []
        
        # Enganchamos los hooks a la capa específica
        self.handlers.append(target_layer.register_forward_hook(self.save_feature_maps))
        self.handlers.append(target_layer.register_full_backward_hook(self.save_gradients))

    def save_feature_maps(self, module, input, output):
        # .detach().clone() es VITAL para evitar errores en DenseNet
        self.feature_maps = output.detach().clone()

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach().clone()

    def generate_cam(self, input_tensor, target_class=None):
        # 1. Forward
        input_tensor.requires_grad = True
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        # 2. Backward
        self.model.zero_grad()
        one_hot_output = torch.zeros_like(output)
        one_hot_output[0][target_class] = 1
        
        output.backward(gradient=one_hot_output, retain_graph=True)
        
        # 3. Generar Heatmap
        gradients = self.gradients
        activations = self.feature_maps
        
        # Promedio global de gradientes (Pesos de importancia)
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        
        # Ponderación de los mapas de características
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        # Promedio de canales para obtener una sola imagen 2D
        heatmap = torch.mean(activations, dim=1).squeeze()
        
        # ReLU (Solo nos importa la contribución positiva)
        heatmap = F.relu(heatmap)
        
        # Normalizar entre 0 y 1
        if torch.max(heatmap) > 0:
            heatmap /= torch.max(heatmap)
        
        return heatmap.cpu().numpy(), target_class, output

    def remove_hooks(self):
        for handle in self.handlers:
            handle.remove()