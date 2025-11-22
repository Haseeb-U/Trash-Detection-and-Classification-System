"""
Intelligent Waste Detection and Classification System
Using YOLO11 for Real-time Trash Detection

Author: Deep Learning Project
Date: November 2025
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from pathlib import Path
import pandas as pd
import time
from ultralytics import YOLO
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Trash Detection System",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        color: #FFFFFF;
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #43A047 0%, #2E7D32 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    .sub-header {
        font-size: 1.5rem;
        color: var(--text-color);
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: var(--secondary-background-color);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        border: 1px solid rgba(128, 128, 128, 0.1);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.2);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4CAF50;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 1rem;
        color: var(--text-color);
        opacity: 0.8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
    div[data-testid="stExpander"] {
        border: 1px solid rgba(128, 128, 128, 0.1);
        border-radius: 10px;
        background-color: var(--secondary-background-color);
    }
    </style>
""", unsafe_allow_html=True)

# Constants
MODEL_PATH = "best_trash_detector.pt"
CONFIDENCE_THRESHOLD = 0.25
CLASS_NAMES = ['Glass', 'Metal', 'Paper', 'Plastic', 'Waste']

# Recyclable classification
RECYCLABLE_CLASSES = {
    'Glass', 'Metal', 'Paper', 'Plastic'
}

@st.cache_resource
def load_model():
    """Load YOLO model with caching"""
    try:
        if os.path.exists(MODEL_PATH):
            model = YOLO(MODEL_PATH)
            return model
        else:
            st.error(f"Model file not found: {MODEL_PATH}")
            st.info("Please train the model first using the Jupyter notebook.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def process_image(image, model, conf_threshold):
    """Process image and return detections"""
    # Run inference
    results = model.predict(
        source=image,
        conf=conf_threshold,
        save=False,
        verbose=False
    )

    # Get annotated image
    annotated_img = results[0].plot()
    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    # Extract detection information
    boxes = results[0].boxes
    detections = []

    for box in boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = CLASS_NAMES[class_id]

        detections.append({
            'Class': class_name,
            'Confidence': confidence,
            'Recyclable': 'Yes' if class_name in RECYCLABLE_CLASSES else 'No'
        })

    return annotated_img, detections

def create_detection_stats(detections):
    """Create statistics from detections"""
    if not detections:
        return 0, 0, 0, 0.0

    df = pd.DataFrame(detections)

    total_detections = len(detections)
    recyclable_count = sum(1 for d in detections if d['Recyclable'] == 'Yes')
    non_recyclable_count = total_detections - recyclable_count
    avg_confidence = df['Confidence'].mean()

    return total_detections, recyclable_count, non_recyclable_count, avg_confidence

def create_class_distribution_chart(detections):
    """Create a bar chart of detected classes"""
    if not detections:
        return None

    df = pd.DataFrame(detections)
    class_counts = df['Class'].value_counts()

    fig = px.bar(
        x=class_counts.index,
        y=class_counts.values,
        labels={'x': 'Waste Class', 'y': 'Count'},
        title='Detected Waste Classes Distribution',
        color=class_counts.values,
        color_continuous_scale='Greens'
    )
    fig.update_layout(showlegend=False, height=400)

    return fig

def create_recyclability_chart(detections):
    """Create a pie chart for recyclable vs non-recyclable"""
    if not detections:
        return None

    df = pd.DataFrame(detections)
    recyclable_counts = df['Recyclable'].value_counts()

    fig = go.Figure(data=[go.Pie(
        labels=['Recyclable', 'Non-Recyclable'],
        values=[recyclable_counts.get('Yes', 0), recyclable_counts.get('No', 0)],
        marker=dict(colors=['#4CAF50', '#FF9800'])
    )])
    fig.update_layout(title='Recyclability Distribution', height=400)

    return fig

def display_metric_card(label, value, icon=""):
    """Display a custom styled metric card"""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{icon} {label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">♻️ Intelligent Waste Detection System</h1>', 
                unsafe_allow_html=True)

    # Load model
    model = load_model()

    if model is None:
        st.stop()

    # Sidebar
    st.sidebar.title("⚙️ Settings")
    st.sidebar.markdown("---")

    # Confidence threshold
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="Minimum confidence for detections"
    )

    # Detection mode
    st.sidebar.markdown("### 📷 Detection Mode")
    mode = st.sidebar.selectbox(
        "Choose detection mode:",
        ["📷 Image Upload", "📁 Batch Processing"],
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")

    # Information section
    with st.sidebar.expander("ℹ️ About This System"):
        st.markdown("""
        **Trash Detection & Classification**

        This system uses YOLO11 to detect and classify 5 different types of waste items.

        **Features:**
        - Real-time object detection
        - Recyclability classification
        - Detailed statistics
        - Batch processing support

        **Model:** YOLO11 trained on a custom Roboflow dataset.
        """)

    with st.sidebar.expander("📋 Detected Classes"):
        for i, class_name in enumerate(CLASS_NAMES, 1):
            recyclable = "♻️" if class_name in RECYCLABLE_CLASSES else "🗑️"
            st.text(f"{recyclable} {class_name}")

    # Main content area
    st.markdown("---")

    # Mode: Image Upload
    if mode == "📷 Image Upload":
        st.subheader("📷 Upload Image for Detection")

        col1, col2 = st.columns([1, 1])

        with col1:
            uploaded_file = st.file_uploader(
                "Choose an image...",
                type=['jpg', 'jpeg', 'png', 'webp'],
                help="Upload an image containing waste items"
            )

            if uploaded_file is not None:
                # Load image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", width="stretch")

                # Process button
                if st.button("🔍 Detect Trash", type="primary", width="stretch"):
                    with st.spinner("Analyzing image..."):
                        # Convert PIL to numpy
                        img_array = np.array(image)

                        # Process image
                        annotated_img, detections = process_image(
                            img_array, model, conf_threshold
                        )

                        # Store results in session state
                        st.session_state['annotated_img'] = annotated_img
                        st.session_state['detections'] = detections

        with col2:
            if 'annotated_img' in st.session_state and 'detections' in st.session_state:
                st.image(
                    st.session_state['annotated_img'],
                    caption="Detection Results",
                    width="stretch"
                )

        # Display results
        if 'detections' in st.session_state:
            detections = st.session_state['detections']

            st.markdown("---")
            st.subheader("📊 Detection Summary")

            # Statistics
            total, recyclable, non_recyclable, avg_conf = create_detection_stats(detections)

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                display_metric_card("Total Detections", total, "📦")
            with col2:
                display_metric_card("Recyclable", recyclable, "♻️")
            with col3:
                display_metric_card("Non-Recyclable", non_recyclable, "🗑️")
            with col4:
                display_metric_card("Avg Confidence", f"{avg_conf:.2%}", "🎯")

            # Charts
            if detections:
                st.markdown("---")
                col1, col2 = st.columns(2)

                with col1:
                    chart1 = create_class_distribution_chart(detections)
                    if chart1:
                        st.plotly_chart(chart1, width="stretch")

                with col2:
                    chart2 = create_recyclability_chart(detections)
                    if chart2:
                        st.plotly_chart(chart2, width="stretch")

                # Detailed detections table
                st.markdown("---")
                st.subheader("🔎 Detailed Detection Results")

                df = pd.DataFrame(detections)
                df['Confidence'] = df['Confidence'].apply(lambda x: f"{x:.2%}")
                df.index = range(1, len(df) + 1)

                st.dataframe(
                    df,
                    use_container_width=True,
                    height=min(400, 50 + len(df) * 35)
                )

    # Mode: Batch Processing
    elif mode == "📁 Batch Processing":
        st.subheader("📁 Batch Image Processing")

        uploaded_files = st.file_uploader(
            "Choose multiple images...",
            type=['jpg', 'jpeg', 'png', 'webp'],
            accept_multiple_files=True,
            help="Upload multiple images for batch processing"
        )

        if uploaded_files:
            st.info(f"📊 {len(uploaded_files)} images uploaded")

            if st.button("🚀 Process All Images", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()

                all_results = []

                for idx, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}")

                    # Load and process image
                    image = Image.open(uploaded_file)
                    img_array = np.array(image)

                    annotated_img, detections = process_image(
                        img_array, model, conf_threshold
                    )

                    all_results.append({
                        'filename': uploaded_file.name,
                        'image': image,
                        'annotated': annotated_img,
                        'detections': detections
                    })

                    progress_bar.progress((idx + 1) / len(uploaded_files))

                status_text.text("✅ All images processed!")

                # Display results
                st.markdown("---")
                st.subheader("📊 Batch Processing Results")

                # Overall statistics
                all_detections = []
                for result in all_results:
                    all_detections.extend(result['detections'])

                total, recyclable, non_recyclable, avg_conf = create_detection_stats(all_detections)

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    display_metric_card("Total Detections", total, "📦")
                with col2:
                    display_metric_card("Recyclable", recyclable, "♻️")
                with col3:
                    display_metric_card("Non-Recyclable", non_recyclable, "🗑️")
                with col4:
                    display_metric_card("Avg Confidence", f"{avg_conf:.2%}", "🎯")

                # Individual results
                st.markdown("---")
                st.subheader("🖼️ Individual Results")

                for result in all_results:
                    with st.expander(f"📷 {result['filename']} - {len(result['detections'])} detections"):
                        col1, col2 = st.columns(2)

                        with col1:
                            st.image(result['image'], caption="Original", width="stretch")

                        with col2:
                            st.image(result['annotated'], caption="Detected", width="stretch")

                        if result['detections']:
                            df = pd.DataFrame(result['detections'])
                            df['Confidence'] = df['Confidence'].apply(lambda x: f"{x:.2%}")
                            st.dataframe(df, width="stretch")

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666;'>
            <p>🌍 Intelligent Waste Detection System | YOLO11 | Deep Learning Project 2025</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
