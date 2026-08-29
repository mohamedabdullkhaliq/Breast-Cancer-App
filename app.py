import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os

st.set_page_config(page_title="Breast Cancer Detection", layout="wide")

# الترويسة الأكاديمية
col1, col2 = st.columns([1, 4])
with col1:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=150)
with col2:
    st.markdown("### كلية الجزيرة التقنية")
    st.markdown("####  Gezira College of Technology")

st.markdown("---")
st.title("🩺 Breast Cancer Detection & Classification System")
st.markdown("**مشرف البحث:** د. سماح قيصر")
st.markdown("**إعداد:** 1/ الصديق حسين علي - 2/ عبدالسلام شحات احمد - 3/ محمد الدومه حسين")
st.markdown("---")

# تحميل النماذج
@st.cache_resource
def load_us_model():
    return load_model('breast_cancer_model.keras', compile=False)

@st.cache_resource
def load_mg_model():
    return load_model('mammogram_model.keras', compile=False)

# اختيار نوع الأشعة
scan_type = st.radio("Select Scan Type (اختر نوع الأشعة):", 
                     ["Ultrasound (موجات فوق صوتية)", "Mammogram (أشعة سينية - ماموجرام)"])

active_model = None
try:
    if scan_type == "Ultrasound (موجات فوق صوتية)":
        active_model = load_us_model()
        classes = ["Normal (أنسجة سليمة)", "Benign (ورم حميد)", "Malignant (ورم خبيث)"]
    else:
        active_model = load_mg_model()
        classes = ["Benign (ورم حميد)", "Malignant (ورم خبيث)"]
        
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"⚠️ جاري تجهيز الذكاء الاصطناعي أو الملف غير موجود ({e})")

# رفع الصورة وتحليلها
if model_loaded:
    uploaded_file = st.file_uploader(f"Upload a {scan_type.split(' ')[0]} image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        col_img, col_res = st.columns(2)
        with col_img:
            st.image(image, caption="Uploaded Scan", use_container_width=True)
            
        with col_res:
            st.write("### 🔍 Analysis in progress...")
            img_array = np.array(image)
            resized_img = cv2.resize(img_array, (128, 128))
            normalized_img = resized_img / 255.0
            reshaped_img = np.reshape(normalized_img, (1, 128, 128, 3))
            
            prediction = active_model.predict(reshaped_img)
            class_index = np.argmax(prediction)
            
            result = classes[class_index]
            
            st.success(f"## Diagnosis: \n### {result}")