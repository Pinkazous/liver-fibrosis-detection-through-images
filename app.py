import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import io

# Importar tus módulos locales
from model_architecture import TextureNetwork
from utils import LABELS
from gradcam import GradCAM
from torchvision import transforms

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Detección Fibrosis Hepática",
    layout="wide"
)

# --- CONSTANTES ---
MODEL_PATH = 'best_fibrosis_finaldensenet.pth' # Asegúrate que este sea el nombre correcto
DEVICE = torch.device("cpu") 
IMG_SIZE = 224
LABELS = {
    0: 'F0 (Sano)',
    1: 'F1 (Fibrosis Leve)',
    2: 'F2 (Fibrosis Moderada)',
    3: 'F3 (Fibrosis Severa)',
    4: 'F4 (Cirrosis)'
}

# --- FUNCIONES AUXILIARES ---

@st.cache_resource
def load_model():
    """Cargar modelo en caché."""
    try:
        # Instanciar arquitectura
        model = TextureNetwork(model_type='densenet', num_classes=5)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error cargando el modelo: {e}")
        return None

def process_uploaded_image(uploaded_file):
    """Convierte el archivo de formato OpenCV/Tensor."""
    # Leer bytes
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1) # Imagen a color original
    
    # Preprocesamiento
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Recorte y filtros
    _, thresh = cv2.threshold(img_gray, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        m = int(w * 0.05)
        if w > 2*m and h > 2*m:
            img_gray = img_gray[y+m : y+h-m, x+m : x+w-m]
            
    img_gray = cv2.medianBlur(img_gray, 5)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_gray = clahe.apply(img_gray)
    
    # Preparar para Tensor
    img_rgb_proc = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    img_resized = cv2.resize(img_rgb_proc, (IMG_SIZE, IMG_SIZE))
    
    img_float = img_resized.astype(np.float32) / 255.0
    img_transposed = img_float.transpose(2, 0, 1)
    tensor = torch.tensor(img_transposed, dtype=torch.float32)
    
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tensor = normalize(tensor).unsqueeze(0) # Batch dim
    
    return tensor, img_bgr 

# --- INTERFAZ DE USUARIO ---

st.write("## Proyecto final análisis con deep learning")
st.write("# Detección de fibrosis hepática a través de imagenes")
st.markdown("Modelo capaz de detectar patrones de imágenes de ultrasonido modo-B para clasificar estadios de fibrosis ($F0$ a $F4$).**")

# 1. BOTÓN Carga
uploaded_file = st.sidebar.file_uploader("Cargar Imagen (.jpg, .png)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Procesar imagen al instante
    tensor_img, original_cv2 = process_uploaded_image(uploaded_file)
    
    # Mostrar imagen original en sidebar
    st.sidebar.image(uploaded_file, caption="Imagen Cargada", width='stretch')
    
    # Cargar modelo
    model = load_model()

    # Contenedor principal
    col1, col2 = st.columns(2)

    # 2. BOTÓN PREDICCIÓN
    if st.sidebar.button("Realizar Predicción"):
        with st.spinner('Analizando...'):
            with torch.no_grad():
                output = model(tensor_img)
                probs = torch.nn.functional.softmax(output, dim=1)
                score, idx = torch.max(probs, 1)
                
            prediction = LABELS[idx.item()]
            confidence = score.item() * 100
            
            # Guardar en estado de sesión
            st.session_state['pred'] = prediction
            st.session_state['conf'] = confidence
            st.session_state['probs'] = probs.numpy()[0]

    # Mostrar resultado si existe en sesión
    if 'pred' in st.session_state:
        st.info(f"### Diagnóstico: **{st.session_state['pred']}**")
        st.progress(int(st.session_state['conf']))
        st.write(f"Confianza del modelo: {st.session_state['conf']:.2f}%")

    # 3. BOTÓN GENERAR GRAD-CAM
    if st.sidebar.button("Generar análisis visual"):
        if model:
            with st.spinner('Generando mapa de calor explicativo...'):
                # Configurar GradCAM
                target_layer = model.features.norm5
                cam = GradCAM(model, target_layer)
                
                # Generar
                heatmap, pred_idx, _ = cam.generate_cam(tensor_img)
                cam.remove_hooks() # Limpieza importante
                
                # Visualización
                # Redimensionar heatmap al tamaño de la imagen original cargada
                h, w = original_cv2.shape[:2]
                heatmap_resized = cv2.resize(heatmap, (w, h))
                
                heatmap_uint8 = np.uint8(255 * heatmap_resized)
                heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                
                # Superponer
                overlay = heatmap_colored * 0.4 + original_cv2 * 0.6
                overlay = np.uint8(overlay)
                
                # Convertir BGR a RGB para Streamlit/PIL
                overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                original_rgb = cv2.cvtColor(original_cv2, cv2.COLOR_BGR2RGB)
                
                # Guardar en sesión
                st.session_state['gradcam_img'] = overlay_rgb
                st.session_state['original_rgb'] = original_rgb

    # Mostrar Comparativa Lado a Lado
    if 'gradcam_img' in st.session_state:
        with col1:
            st.subheader("Imagen Original")
            st.image(st.session_state['original_rgb'], width='stretch')
        
        with col2:
            st.subheader("Mapa de calor Grad Cam")
            st.image(st.session_state['gradcam_img'], width='stretch')
            
        # 4. BOTÓN DESCARGAR
        # Convertir array numpy a bytes para descarga
        result_pil = Image.fromarray(st.session_state['gradcam_img'])
        buf = io.BytesIO()
        result_pil.save(buf, format="JPEG")
        byte_im = buf.getvalue()

        st.download_button(
            label="⬇️ Descargar Análisis Grad-CAM",
            data=byte_im,
            file_name="diagnostico_gradcam.jpg",
            mime="image/jpeg"
        )
