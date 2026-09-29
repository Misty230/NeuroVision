import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="NeuroVision",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background:
            radial-gradient(circle at top left, #dbeafe 0%, transparent 35%),
            radial-gradient(circle at top right, #ede9fe 0%, transparent 35%),
            linear-gradient(135deg, #f8fbff 0%, #eef4ff 100%);
    }

    /* Hide Streamlit default elements */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    /* Main container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1150px;
    }

    /* Hero section */
    .hero {
        background: linear-gradient(
            135deg,
            #2563eb 0%,
            #4f46e5 50%,
            #7c3aed 100%
        );

        padding: 45px 40px;
        border-radius: 28px;
        color: white;
        text-align: center;

        box-shadow:
            0 20px 45px rgba(37, 99, 235, 0.25);

        margin-bottom: 30px;
    }

    .hero h1 {
        font-size: 52px;
        font-weight: 800;
        margin-bottom: 8px;
        letter-spacing: -1px;
    }

    .hero h2 {
        font-size: 24px;
        font-weight: 500;
        margin-bottom: 15px;
    }

    .hero p {
        font-size: 17px;
        opacity: 0.92;
    }

    /* Section title */
    .section-title {
        color: #172554;
        font-size: 27px;
        font-weight: 750;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    /* Upload card */
    .upload-card {
        background: white;
        border-radius: 22px;
        padding: 25px;
        box-shadow: 0 8px 30px rgba(30, 64, 175, 0.10);
        border: 1px solid #dbeafe;
        margin-bottom: 25px;
    }

    /* Result cards */
    .result-no-tumor {
        background: linear-gradient(135deg, #dcfce7, #bbf7d0);
        border: 2px solid #22c55e;
        border-radius: 22px;
        padding: 28px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(34, 197, 94, 0.15);
    }

    .result-tumor {
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        border: 2px solid #ef4444;
        border-radius: 22px;
        padding: 28px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(239, 68, 68, 0.15);
    }

    .result-icon {
        font-size: 48px;
    }

    .result-title {
        font-size: 30px;
        font-weight: 800;
        margin: 8px 0;
    }

    .result-subtitle {
        font-size: 18px;
        font-weight: 600;
    }

    /* Confidence card */
    .confidence-card {
        background: white;
        border-radius: 20px;
        padding: 22px;
        text-align: center;
        border: 1px solid #e0e7ff;
        box-shadow: 0 8px 25px rgba(30, 64, 175, 0.08);
        margin-top: 20px;
    }

    .confidence-number {
        font-size: 38px;
        font-weight: 800;
        color: #4f46e5;
    }

    .confidence-label {
        font-size: 15px;
        color: #64748b;
    }

    /* Info cards */
    .info-card {
        background: white;
        padding: 22px;
        border-radius: 18px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 6px 20px rgba(15, 23, 42, 0.06);
        height: 100%;
    }

    .info-card h3 {
        color: #1e3a8a;
        margin-bottom: 8px;
    }

    .info-card p {
        color: #64748b;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        border-radius: 14px;
        height: 52px;

        background: linear-gradient(
            90deg,
            #2563eb,
            #7c3aed
        );

        color: white;
        font-size: 18px;
        font-weight: 700;

        border: none;

        box-shadow:
            0 8px 20px rgba(79, 70, 229, 0.25);

        transition: 0.2s;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow:
            0 12px 25px rgba(79, 70, 229, 0.35);
    }

    /* Probability labels */
    .prob-title {
        font-size: 17px;
        font-weight: 700;
        color: #1e293b;
        margin-top: 12px;
    }

    /* Disclaimer */
    .disclaimer {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        border-radius: 16px;
        padding: 18px;
        color: #9a3412;
        margin-top: 30px;
        font-size: 14px;
    }

    /* Footer */
    .custom-footer {
        text-align: center;
        color: #64748b;
        margin-top: 35px;
        font-size: 14px;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# MODEL
# =========================================================

class_names = [
    "glioma",
    "meningioma",
    "pituitary",
    "notumor"
]

MODEL_PATH = "neurovision_resnet50.keras"


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()


# =========================================================
# PREDICTION
# =========================================================

def predict_mri(image):

    image = image.convert("RGB")

    image = image.resize((224, 224))

    image_array = np.array(image).astype("float32")

    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = np.argmax(predictions)

    predicted_class = class_names[predicted_index]

    confidence = predictions[predicted_index] * 100

    return predicted_class, confidence, predictions


# =========================================================
# HERO
# =========================================================

st.markdown("""
<div class="hero">

    <div style="font-size:60px;">🧠</div>

    <h1>NeuroVision</h1>

    <h2>Brain Tumor Detection & Classification</h2>

    <p>
        AI-assisted MRI image analysis using Deep Learning
    </p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# INTRO
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-card">
        <h3>🧠 AI Powered</h3>
        <p>
        Uses a trained ResNet50 deep learning model
        for MRI image classification.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <h3>🔬 MRI Analysis</h3>
        <p>
        Upload an MRI image and receive an
        AI-assisted classification result.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-card">
        <h3>📊 4 Classes</h3>
        <p>
        Glioma, Meningioma, Pituitary and
        No Tumor.
        </p>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# UPLOAD SECTION
# =========================================================

st.markdown(
    '<div class="section-title">📤 Upload MRI Image</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    left, right = st.columns([1.2, 1])

    # -----------------------------------------------------
    # IMAGE
    # -----------------------------------------------------

    with left:

        st.markdown(
            '<div class="section-title">🖼️ MRI Preview</div>',
            unsafe_allow_html=True
        )

        st.image(
            image,
            use_container_width=True
        )


    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    with right:

        st.markdown(
            '<div class="section-title">🔍 Analysis</div>',
            unsafe_allow_html=True
        )

        if st.button(
            "🔍 Analyze MRI",
            use_container_width=True
        ):

            with st.spinner("Analyzing MRI image..."):

                predicted_class, confidence, predictions = predict_mri(
                    image
                )

            # -------------------------------------------------
            # RESULT
            # -------------------------------------------------

            if predicted_class == "notumor":

                st.markdown(f"""
                <div class="result-no-tumor">

                    <div class="result-icon">🟢</div>

                    <div class="result-title">
                        No Tumor Predicted
                    </div>

                    <div class="result-subtitle">
                        The model predicted: No Tumor
                    </div>

                </div>
                """, unsafe_allow_html=True)

            else:

                st.markdown(f"""
                <div class="result-tumor">

                    <div class="result-icon">🔴</div>

                    <div class="result-title">
                        Tumor Predicted
                    </div>

                    <div class="result-subtitle">
                        Type: {predicted_class.title()}
                    </div>

                </div>
                """, unsafe_allow_html=True)


            # -------------------------------------------------
            # CONFIDENCE
            # -------------------------------------------------

            st.markdown(f"""
            <div class="confidence-card">

                <div class="confidence-label">
                    MODEL CONFIDENCE
                </div>

                <div class="confidence-number">
                    {confidence:.2f}%
                </div>

            </div>
            """, unsafe_allow_html=True)


# =========================================================
# PROBABILITIES
# =========================================================

if uploaded_file is not None and "predictions" in locals():

    st.markdown(
        '<div class="section-title">📊 Class Probabilities</div>',
        unsafe_allow_html=True
    )

    probability_columns = st.columns(4)

    for i, class_name in enumerate(class_names):

        probability = predictions[i]

        with probability_columns[i]:

            st.markdown(
                f"""
                <div class="info-card">
                    <h3>{class_name.title()}</h3>
                    <p>
                    {probability * 100:.2f}%
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(float(probability))


# =========================================================
# DISCLAIMER
# =========================================================

st.markdown("""
<div class="disclaimer">

    ⚠️ <b>Important:</b>
    NeuroVision is an academic research/demo application.
    It is not a medical diagnostic system and should not be
    used for clinical decisions.

</div>
""", unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="custom-footer">

    🧠 <b>NeuroVision</b> &nbsp; | &nbsp;
    Brain Tumor Detection & Classification

    <br><br>

    Developed as an Academic Machine Learning Project

</div>
""", unsafe_allow_html=True)