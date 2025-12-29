import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tempfile
import warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.applications import MobileNetV2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# 1. SETUP & UI CONFIGURATION
warnings.filterwarnings("ignore")

def set_background(image_file):
    import base64
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{b64_encoded}");
        background-size: cover;
    }}
    </style>
    """
    st.markdown(style, unsafe_allow_html=True)

try:
    set_background('files/background.jpg')
except:
    pass

# 2. MODEL SETUP & CLASSIFICATION LOGIC
class_names = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic']

def classify(image, model, class_names):
    # Standardize image for MobileNetV2 input
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    index = np.argmax(predictions)
    return class_names[index], predictions[0][index]

@st.cache_resource
def get_model():
    # Architecture optimized for 80:10:10 split, 0.001 LR, 32 Batch Size
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers[:-4]:
        layer.trainable = False
        
    model = Sequential([
        base_model,
        Flatten(),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.08),
        Dense(len(class_names), activation='softmax')
    ])
    
    model.load_weights('MobileNetV2_811.keras')
    return model

model = get_model()

# 3. SIDEBAR NAVIGATION
app_mode = st.sidebar.selectbox('Choose the app mode', 
                                ['About', 'Image Upload', 'Video File Upload', 'Real-time Video'])

# --- MODE: ABOUT ---
if app_mode == 'About':
    st.markdown("<h1 style='text-align: center;'>♻️ Welcome to Sortify</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2em;'>Redefining Waste Management with Intelligent Vision.</p>", unsafe_allow_html=True)
    st.divider()

    st.subheader("Our Mission")
    st.write(
        "Sortify is an AI-powered waste classification prototype developed to tackle low public participation "
        "in waste separation at source initiatives in Malaysia. By leveraging Deep Learning, we empower "
        "households to correctly identify recyclables in real-time."
    )

    st.subheader("Model Performance Intelligence")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Testing Accuracy", value="93.06%", delta="MobileNetV2")
    with col2:
        st.metric(label="Lowest Test Loss", value="0.1919", delta="Robust Reliability")
    with col3:
        st.metric(label="Avg F1-Score", value="0.94", delta="Balanced Precision")

    with st.expander("🔍 Explore the Architecture"):
        st.write("Optimized through extensive experimentation against VGG16 and InceptionV3.")
        tech_col1, tech_col2 = st.columns(2)
        with tech_col1:
            st.info("**Dataset Insights**\n- 7,777 Total Images From Kaggle\n- 5 Core Categories: Cardboard, Glass, Metal, Paper, Plastic")
        with tech_col2:
            st.success("**Optimal Hyperparameters**\n- Split: 80:10:10\n- Learning Rate: 0.001\n- Batch Size: 32")

    st.divider()
    st.caption("Sortify - AI-Powered Waste Classification System © 2025")

# --- MODE: IMAGE UPLOAD ---
elif app_mode == 'Image Upload':
    st.header('Image Classification')
    file = st.file_uploader('Upload an image', type=['jpeg', 'jpg', 'png'])
    if file:
        image = Image.open(file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_container_width=True)
        class_name, conf_score = classify(image, model, class_names)
        st.success(f"Prediction: **{class_name}**")
        st.info(f"Confidence Score: {int(conf_score * 100)}%")

# --- MODE: VIDEO FILE UPLOAD ---
elif app_mode == 'Video File Upload':
    st.header('Video File Classification')
    video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)
        frame_window = st.image([])
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            class_name, conf_score = classify(pil_img, model, class_names)
            label = f"{class_name}: {int(conf_score * 100)}%"
            cv2.putText(frame, label, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

# --- MODE: REAL-TIME VIDEO ---
elif app_mode == 'Real-time Video':
    st.header('Real-time Waste Detection')
    class VideoProcessor(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            class_name, conf_score = classify(pil_img, model, class_names)
            label = f"{class_name}: {int(conf_score * 100)}%"
            cv2.putText(img, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return img
    webrtc_streamer(key="waste-classification", video_transformer_factory=VideoProcessor)