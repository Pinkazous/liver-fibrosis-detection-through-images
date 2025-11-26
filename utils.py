# utils.py
import cv2
import numpy as np
import torch
from torchvision import transforms

LABELS = {
    0: 'F0 (Sano)',
    1: 'F1 (Fibrosis Leve)',
    2: 'F2 (Fibrosis Moderada)',
    3: 'F3 (Fibrosis Severa)',
    4: 'F4 (Cirrosis)'
}

def preprocess_ultrasound_inference(img_path, img_size=224):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    
    # 1. Recorte y Limpieza (ROI)
    _, thresh = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        m = int(w * 0.05)
        if w > 2*m and h > 2*m:
            img = img[y+m : y+h-m, x+m : x+w-m]
    if img.size == 0: img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # 2. Filtros de Textura
    img = cv2.medianBlur(img, 5)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)

    # 3. Formato Tensor
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    
    # 4. NORMALIZACIÓN (El paso que arreglamos antes)
    tensor = torch.tensor(img, dtype=torch.float32)
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tensor = normalize(tensor)
    
    return tensor.unsqueeze(0) # Añadir dimensión Batch (1, C, H, W)