import streamlit as st
import pandas as pd
import numpy as np
import cv2
import torch
import io
import joblib
from PIL import Image
from torchvision import transforms
import tensorflow as tf

# --- TUS MÓDULOS LOCALES ---
from model_architecture import TextureNetwork
from gradcam import GradCAM
from utils import LABELS 

LABELS_tabular = {
    0: '(F0-F2) Bajo Riesgo de Fibrosis',
    1: '(F3-F4) Alto Riesgo de Fibrosis',
}


# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Fibrosis AI Multimodal",
    layout="wide"
)

# --- RUTAS DE MODELOS ---
IMG_MODEL_PATH = 'best_fibrosis_finaldensenet.pth'
TAB_MODEL_PATH = 'definite_binary_model.h5'
SCALER_PATH =    'scaler.pkl' 

# --- CONSTANTES ---
MODEL_PATH = 'best_fibrosis_finaldensenet.pth' 
DEVICE = torch.device("cpu") 
IMG_SIZE = 224
LABELS = {
    0: 'F0 (Sano)',
    1: 'F1 (Fibrosis Leve)',
    2: 'F2 (Fibrosis Moderada)',
    3: 'F3 (Fibrosis Severa)',
    4: 'F4 (Cirrosis)'
}

DEVICE = torch.device("cpu")

# --- FUNCIONES DE CARGA ---
@st.cache_resource
def load_image_model():
    try:
        model = TextureNetwork(model_type='densenet', num_classes=5)
        model.load_state_dict(torch.load(IMG_MODEL_PATH, map_location=DEVICE))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error cargando modelo de imagen: {e}")
        return None

@st.cache_resource
def load_clinical_model():
    try:
        model = tf.keras.models.load_model(TAB_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except Exception as e:
        return None, None

def process_uploaded_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Recorte y limpieza
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
    
    img_rgb_proc = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    img_resized = cv2.resize(img_rgb_proc, (224, 224))
    
    img_float = img_resized.astype(np.float32) / 255.0
    img_transposed = img_float.transpose(2, 0, 1)
    tensor = torch.tensor(img_transposed, dtype=torch.float32)
    
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tensor = normalize(tensor).unsqueeze(0)
    
    return tensor, img_bgr

# --- INTERFAZ PRINCIPAL ---

st.title("Sistema de Diagnóstico de Estadios de Fibrosis Hepática")

modo = st.sidebar.radio(
    "Selecciona el tipo de análisis:",
    ("Análisis de Imágenes", "Análisis de Datos Clínicos (Laboratorio)")
)

# ==============================================================================
# MODO 1: IMÁGENES
# ==============================================================================
if modo == "Análisis de Imágenes":
    # st.header("Detección de fibrosis hepática a través de imagenes")
    st.markdown("Modelo capaz de detectar patrones de imágenes de ultrasonido modo-B para clasificar estadios de fibrosis ($F0$ a $F4$).")
    st.caption("Por favor cargar una imagen de ultrasonido modo-B en formato .jpg o .png, a continuación pulsar el botón **Realizar predicción**")
    st.caption("Para visualizar la región de interés detectada por el modelo pulsar el botón **Generar Mapa de Calor (Grad-CAM)**")
    uploaded_file = st.file_uploader("Cargar Imagen (.jpg, .png)", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Procesar y guardar en sesión si cambia el archivo
        tensor_img, original_cv2 = process_uploaded_image(uploaded_file)
        # Convertir BGR a RGB para mostrar correctamente en Streamlit
        original_rgb = cv2.cvtColor(original_cv2, cv2.COLOR_BGR2RGB)
        
        img_model = load_image_model()


        # Botones de Acción
        col_btns1, col_btns2 = st.columns(2)
        
        # 1. BOTÓN PREDICCIÓN
        if col_btns1.button("Realizar Predicción", use_container_width=True):
            if img_model:
                with st.spinner('Procesando...'):
                    with torch.no_grad():
                        output = img_model(tensor_img)
                        probs = torch.nn.functional.softmax(output, dim=1)
                        score, idx = torch.max(probs, 1)
                    
                    # Guardar resultado en sesión
                    st.session_state['pred_result'] = LABELS[idx.item()]
                    st.session_state['pred_conf'] = score.item()
        
        # Mostrar Predicción (si existe)
        if 'pred_result' in st.session_state:
            st.success(f"Diagnóstico: **{st.session_state['pred_result']}**")
            st.progress(st.session_state['pred_conf'])
            st.caption(f"Confianza: {st.session_state['pred_conf']*100:.2f}%")
            st.divider()

        # 2. BOTÓN GRAD-CAM
        if col_btns2.button("Generar Mapa de Calor (Grad-CAM)", use_container_width=True):
            if img_model:
                with st.spinner('Analizando zonas de atención...'):
                    target_layer = img_model.features.norm5
                    cam = GradCAM(img_model, target_layer)
                    heatmap, _, _ = cam.generate_cam(tensor_img)
                    cam.remove_hooks()
                    
                    # Crear Overlay
                    h, w = original_cv2.shape[:2]
                    heatmap_resized = cv2.resize(heatmap, (w, h))
                    heatmap_uint8 = np.uint8(255 * heatmap_resized)
                    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                    overlay = heatmap_colored * 0.4 + original_cv2 * 0.6
                    overlay_rgb = cv2.cvtColor(np.uint8(overlay), cv2.COLOR_BGR2RGB)
                    
                    # Guardar imagen generada en sesión
                    st.session_state['gradcam_image'] = overlay_rgb

        # --- SECCIÓN VISUALIZACIÓN COMPARATIVA ---
        if 'gradcam_image' in st.session_state:
            st.subheader("Visualización Explicable")
            
            # Usar columnas para ponerlas lado a lado
            col_orig, col_grad = st.columns(2)
            
            with col_orig:
                st.image(original_rgb, caption="Imagen Original", use_container_width=True)
            
            with col_grad:
                st.image(st.session_state['gradcam_image'], caption="Zonas de Fibrosis (IA)", use_container_width=True)
                
                # --- BOTÓN DE DESCARGA ---
                # Convertir array numpy a bytes para descarga
                gradcam_pil = Image.fromarray(st.session_state['gradcam_image'])
                buf = io.BytesIO()
                gradcam_pil.save(buf, format="JPEG")
                byte_im = buf.getvalue()

                st.download_button(
                    label="Descargar Imagen Grad-CAM",
                    data=byte_im,
                    file_name="diagnostico_gradcam.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )

# ==============================================================================
# MODO 2: CLÍNICO
# ==============================================================================
else:
    
    st.markdown("Estimación de Riesgo por Parámetros Clínicos.")
    keras_model, scaler = load_clinical_model()
    model_artifact=joblib.load('random_forest_artifact.joblib')
    scaler=model_artifact.get('scaler')
    model=model_artifact.get('model')
    label_encoder=model_artifact.get('encoder')
    
    
    
    if model_artifact is None:
        st.error("Error cargando modelo clínico.")
    else:
        tab1, tab2 = st.tabs(["Ingreso Manual", "Carga Masiva"])
        
        with tab1:
            st.markdown("Por favor introducir los siguientes valores y a continuación pulsar el botón **predecir riesgo**")
            st.caption("BILIRRUBINA DIRECTA, BILIRRUBINA INDIRECTA, BILIRRUBINA TOTAL, LDH, FOSFATASA ALCALINA, (GGT), INR, PLAQUETAS, (TGO-AST), (TGP-ALT) ")
           
            c1, c2, c3, c4, c5, c6, c7, c8, c9, c10 = st.columns(10)
            BD = c1.number_input('B.DIRECT',width =100,value=0.1)
            BI = c2.number_input("B.INDIRECT", width =100,value=0.6)
            BT = c3.number_input("B.TOTAL", width =100,value=0.7)
            LDH = c4.number_input("LDH", width =100,value=369)
            FA = c5.number_input("FOSF.ALC",width =100,value=85)
            GGT = c6.number_input("GGT", width =100,value=36)
            INR = c7.number_input("INR", width =100,value=1.07)
            PL = c8.number_input("PLAQUETAS", width =100,value=336)
            TGO_AST = c9.number_input("TGO-AST", width =100,value=28)
            TGP_ALT = c10.number_input("TGP-ALT",width =100,value=19)
                        
            # --------------Predecir-----------------------
            if st.button("Predecir Riesgo"):
                input_data = np.array([[BD, BI, BT, LDH, FA,GGT,INR,PL,TGO_AST,TGP_ALT]])
                scaled_data_=scaler.transform(input_data)
                predicted_val=model.predict(scaled_data_)
                if hasattr(predicted_val, "ndim") and predicted_val.ndim > 1:
                    predicted_idx = np.argmax(predicted_val, axis=1)
                else:
                    predicted_idx = np.array(predicted_val).astype(int)
                    
                # Usar inverse_transform para obtener etiquetas originales
                predicted_label = label_encoder.inverse_transform(predicted_idx)
                st.session_state['pred_result_'] = predicted_label[0]
                if 'pred_result_' in st.session_state:
                    st.success(f"Diagnóstico: **{st.session_state['pred_result_']}**")
                    st.divider()
                
        with tab2:
            st.caption("**IMPORTANTE** El archivo que se cargue debe tener las siguientes columnas:")
            st.caption("BILIRRUBINA DIRECTA, BILIRRUBINA INDIRECTA, BILIRRUBINA TOTAL, LDH, FOSFATASA ALCALINA, (GGT), INR, PLAQUETAS, (TGO-AST), (TGP-ALT) ")
            uploaded_csv = st.file_uploader("tipo CSV:", type=["csv"])
            if uploaded_csv:
                df = pd.read_csv(uploaded_csv)
                required = ['BILIRRUBINA DIRECTA',
                            'BILIRRUBINA INDIRECTA',
                            'BILIRRUBINA TOTAL',
                            'LDH',
                            'FOSFATASA ALCALINA',
                            '(GGT)',
                            'INR',
                            'PLAQUETAS',
                            '(TGO-AST)',
                            '(TGP-ALT)']
                if st.button("Predecir"):
                    data = df[required].values
                    scaled_data_=scaler.transform(data)
                    predicted_val=model.predict(scaled_data_)
                    if hasattr(predicted_val, "ndim") and predicted_val.ndim > 1:
                        predicted_idx = np.argmax(predicted_val, axis=1)
                    else:
                        predicted_idx = np.array(predicted_val).astype(int)
                    
                    # Usar inverse_transform para obtener etiquetas originales
                    predicted_label = label_encoder.inverse_transform(predicted_idx)
                    
                    df['Prediccion'] = predicted_label
                    
                    st.dataframe(df)
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button("Descargar Resultados", csv, "resultados.csv", "text/csv")

