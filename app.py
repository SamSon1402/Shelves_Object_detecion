import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from PIL import Image
import glob
import json
import torch
import kagglehub  # For downloading dataset

# Set page configuration
st.set_page_config(
    page_title="Supermarket Shelf Detector",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load black and white CSS styling
def load_css():
    css = """
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&display=swap');
    
    .title {
        font-family: 'Roboto Mono', monospace;
        color: #FFFFFF;
        text-align: center;
        padding: 10px;
        margin-bottom: 20px;
        font-size: 2.5em;
        border: 4px solid #FFFFFF;
        background-color: #000000;
    }
    
    .subtitle {
        font-family: 'Roboto Mono', monospace;
        color: #FFFFFF;
        font-size: 1.5em;
        margin: 15px 0;
        border-bottom: 3px solid #FFFFFF;
        padding-bottom: 5px;
        background-color: #000000;
    }
    
    .text {
        font-family: 'Roboto Mono', monospace;
        color: #000000;
        line-height: 1.6;
        background-color: #FFFFFF;
        padding: 5px;
    }
    
    .metric-box {
        font-family: 'Roboto Mono', monospace;
        background-color: #000000;
        border: 3px solid #FFFFFF;
        padding: 15px;
        text-align: center;
        color: #FFFFFF;
    }
    
    .metric-value {
        font-size: 2em;
        color: #FFFFFF;
    }
    
    .metric-label {
        font-size: 1.2em;
        color: #FFFFFF;
    }
    
    .stApp {
        background-color: #FFFFFF;
    }
    
    /* Dataframe styling */
    .dataframe {
        font-family: 'Roboto Mono', monospace !important;
        border: 3px solid #000000 !important;
    }
    
    .dataframe th {
        background-color: #000000 !important;
        color: #FFFFFF !important;
        font-size: 1.1em !important;
        text-align: center !important;
    }
    
    .dataframe td {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border: 1px solid #000000 !important;
    }
    """
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# Load CSS
load_css()

# Download Kaggle dataset
def download_kaggle_dataset():
    with st.spinner("Downloading the dataset from Kaggle..."):
        try:
            # Download dataset
            path = kagglehub.dataset_download("humansintheloop/supermarket-shelves-dataset")
            st.success(f"Dataset downloaded successfully to: {path}")
            return path
        except Exception as e:
            st.error(f"Error downloading dataset: {e}")
            st.info("If you already have the dataset, you can specify the path manually.")
            return None

# Install and import YOLOv8
def install_dependencies():
    with st.spinner("Installing required packages..."):
        os.system("pip install ultralytics")
        # Force reload to ensure the module is available
        import importlib
        import sys
        if "ultralytics" in sys.modules:
            importlib.reload(sys.modules["ultralytics"])
    st.success("Dependencies installed!")

# Load trained YOLOv8 model
@st.cache_resource
def load_yolo_model(model_path="supermarket_detector.pt"):
    try:
        from ultralytics import YOLO
        # Try to load the custom model
        if os.path.exists(model_path):
            model = YOLO(model_path)
            st.success(f"Successfully loaded custom model from {model_path}")
            # Print model classes for debugging
            st.info(f"Model classes: {model.names}")
            return model
        else:
            st.warning(f"Custom model not found at {model_path}. Loading default YOLOv8 model...")
            model = YOLO("yolov8n.pt")  # Load the default model as fallback
            st.info(f"Default model classes: {model.names}")
            return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

# Process image with YOLOv8 - return the processed image and detections
def process_with_yolo(image, model_path="supermarket_detector.pt", confidence_threshold=0.25):
    try:
        from ultralytics import YOLO
        model = load_yolo_model(model_path)
        
        if model is None:
            st.error("Failed to load the YOLO model.")
            return image, []
        
        # Convert PIL image to numpy array
        img_np = np.array(image)
        
        # Add debug info
        st.write(f"Processing image of shape {img_np.shape} with confidence threshold {confidence_threshold}")
        
        # Run inference
        results = model(img_np, conf=confidence_threshold)
        
        # Process results
        detections = []
        
        # Convert results to our format
        for result in results:
            # Add debug info about boxes
            st.write(f"Detection results: {len(result.boxes)} boxes found")
            
            for box in result.boxes:
                cls_id = int(box.cls.item())
                conf = box.conf.item()
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                detections.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "class": model.names[cls_id],
                    "confidence": conf,
                })
        
        # Draw bounding boxes on image
        annotated_img = results[0].plot(line_width=2, font_size=1, pil=True)
        
        # Convert back to PIL
        result_img = Image.fromarray(np.array(annotated_img))
        
        return result_img, detections
    except Exception as e:
        st.error(f"Error in YOLO processing: {e}")
        return image, []

# Function to calculate basic image statistics
def analyze_image(image, detections=None):
    img_np = np.array(image)
    
    # Convert to grayscale for analysis
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np
    
    # Basic statistics
    brightness = np.mean(gray)
    contrast = np.std(gray)
    
    # Edge detection for shelf structure analysis
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    
    # Detection statistics
    total_detections = 0
    price_count = 0
    product_count = 0
    
    if detections:
        total_detections = len(detections)
        
        # Count by class
        for d in detections:
            class_name = d["class"].lower()
            if "price" in class_name:
                price_count += 1
            elif "product" in class_name:
                product_count += 1
    
    return {
        "brightness": brightness,
        "contrast": contrast,
        "edge_density": edge_density,
        "total_detections": total_detections,
        "price_count": price_count,
        "product_count": product_count
    }

# Function to display image with contours (as a simple detection alternative)
def show_contour_detection(image):
    img_np = np.array(image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Get edges
    edges = cv2.Canny(gray, 50, 150)
    
    # Threshold the image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter by size
    min_contour_area = 500
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    
    # Draw contours on original image
    result = img_np.copy()
    cv2.drawContours(result, filtered_contours, -1, (0, 0, 0), 2)
    
    return Image.fromarray(result)

# Function to find image files recursively with multiple extensions
def find_images(base_path):
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        # Use glob to find files recursively
        pattern = os.path.join(base_path, '**', ext)
        image_files.extend(glob.glob(pattern, recursive=True))
    
    # Sort files for consistent display
    return sorted(image_files)

# Main application
def main():
    # Application title
    st.markdown('<div class="title">Supermarket Shelf Detector</div>', unsafe_allow_html=True)

    # Introduction
    st.markdown('<div class="text">A tool for detecting and analyzing products and price tags on supermarket shelves using our custom-trained YOLOv8 model.</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown('<div class="subtitle">Controls</div>', unsafe_allow_html=True)
        
        # Option to use sample images or upload your own
        input_option = st.radio(
            "Input Source",
            ["Kaggle Dataset", "Upload Image"]
        )
        
        # Path to the custom model
        st.markdown('<div class="subtitle">Model Settings</div>', unsafe_allow_html=True)
        model_path = st.text_input("Model Path", value="supermarket_detector.pt")
        
        # Detection method
        detection_method = st.radio(
            "Detection Method",
            ["YOLOv8 Detection", "Basic Contour Detection"],
            index=0  # Default to YOLO
        )
        
        # Confidence threshold for YOLO
        confidence_threshold = st.slider(
            "YOLOv8 Confidence Threshold", 
            min_value=0.1, 
            max_value=0.9, 
            value=0.25, 
            step=0.05,
            disabled=(detection_method != "YOLOv8 Detection")
        )
        
        show_metrics = st.checkbox("Show Image Metrics", value=True)
        
        # Dataset download option
        st.markdown('<div class="subtitle">Dataset</div>', unsafe_allow_html=True)
        download_dataset = st.button("Download Kaggle Dataset")
        
        if download_dataset:
            dataset_path = download_kaggle_dataset()
            if dataset_path:
                st.session_state['kaggle_dataset_path'] = dataset_path

    # Main area
    if input_option == "Kaggle Dataset":
        st.markdown('<div class="subtitle">Kaggle Dataset Explorer</div>', unsafe_allow_html=True)
        
        # Use downloaded path if available in session state
        default_path = st.session_state.get('kaggle_dataset_path', "/kaggle/input/supermarket-shelves-dataset")
        
        st.markdown('<div class="text">Please specify the path to your dataset files:</div>', unsafe_allow_html=True)
        dataset_path = st.text_input("Dataset Path", value=default_path)
        
        # Option to download dataset if path doesn't exist
        if not os.path.exists(dataset_path):
            st.warning(f"The path '{dataset_path}' does not exist.")
            
            if st.button("Download Dataset Now"):
                new_path = download_kaggle_dataset()
                if new_path:
                    dataset_path = new_path
                    st.session_state['kaggle_dataset_path'] = new_path
                    st.success(f"Using downloaded dataset at: {new_path}")
                    st.experimental_rerun()
            
            # Try to find a suitable path automatically
            possible_paths = [
                "/root/.cache/kagglehub/datasets/humansintheloop/supermarket-shelves-dataset/versions/2",
                "/root/.cache/kagglehub/datasets/humansintheloop/supermarket-shelves-dataset",
                "/kaggle/input/supermarket-shelves-dataset",
                "./supermarket-shelves-dataset"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    st.info(f"Try using this path instead: {path}")
                    dataset_path = path
                    break
        
        st.markdown('<div class="text">Searching for images...</div>', unsafe_allow_html=True)
        
        # Find all images in the dataset
        image_files = find_images(dataset_path)
        
        if image_files:
            st.success(f"Found {len(image_files)} images in the dataset")
            
            # Create a selectbox to choose an image
            selected_image = st.selectbox("Select an image to analyze", image_files)
            
            if selected_image:
                # Load and display the image
                try:
                    image = Image.open(selected_image).convert("RGB")
                    
                    # Show original image and processed image side by side
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown('<div class="text">Original Image</div>', unsafe_allow_html=True)
                        st.image(image, use_container_width=True)
                    
                    with col2:
                        st.markdown('<div class="text">Processed Image</div>', unsafe_allow_html=True)
                        
                        with st.spinner("Processing image..."):
                            if detection_method == "YOLOv8 Detection":
                                # Process with YOLO
                                processed_img, detections = process_with_yolo(image, model_path, confidence_threshold)
                                st.image(processed_img, use_container_width=True)
                                
                                # Show detection summary
                                if detections:
                                    st.markdown(f'<div class="text">YOLOv8 detected {len(detections)} objects</div>', unsafe_allow_html=True)
                                    
                                    # Create a DataFrame for the detections
                                    detection_df = pd.DataFrame([
                                        {
                                            "Class": d["class"], 
                                            "Confidence": f"{d['confidence']*100:.1f}%",
                                            "Position": f"[{d['bbox'][0]}, {d['bbox'][1]}, {d['bbox'][2]}, {d['bbox'][3]}]"
                                        }
                                        for d in detections
                                    ])
                                    
                                    # Display the detection table
                                    st.dataframe(detection_df, use_container_width=True)
                                    
                                    # Summarize by class
                                    class_counts = {}
                                    for d in detections:
                                        cls = d["class"]
                                        class_counts[cls] = class_counts.get(cls, 0) + 1
                                    
                                    # Display summary
                                    st.markdown('<div class="text">Detected Objects:</div>', unsafe_allow_html=True)
                                    for cls, count in class_counts.items():
                                        st.markdown(f'<div class="text">• {cls}: {count}</div>', unsafe_allow_html=True)
                                else:
                                    st.warning("No objects detected in this image.")
                            
                            elif detection_method == "Basic Contour Detection":
                                # Show image with contour detection
                                processed_img = show_contour_detection(image)
                                st.image(processed_img, use_column_width=True)
                                st.info("Basic contour detection is shown. This is not using the YOLO model.")
                    
                    # Image analysis and metrics
                    if show_metrics:
                        st.markdown('<div class="subtitle">Image Analysis</div>', unsafe_allow_html=True)
                        
                        with st.spinner("Analyzing image..."):
                            if detection_method == "YOLOv8 Detection" and 'detections' in locals():
                                metrics = analyze_image(image, detections)
                            else:
                                metrics = analyze_image(image)
                        
                        # Display metrics in a nice layout
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.markdown(f"""
                            <div class="metric-box">
                                <div class="metric-label">BRIGHTNESS</div>
                                <div class="metric-value">{metrics['brightness']:.1f}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div class="metric-box">
                                <div class="metric-label">CONTRAST</div>
                                <div class="metric-value">{metrics['contrast']:.1f}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(f"""
                            <div class="metric-box">
                                <div class="metric-label">PRODUCTS</div>
                                <div class="metric-value">{metrics.get('product_count', 0)}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        with col4:
                            st.markdown(f"""
                            <div class="metric-box">
                                <div class="metric-label">PRICES</div>
                                <div class="metric-value">{metrics.get('price_count', 0)}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # If detections exist, show a pie chart of detections by class
                        if detection_method == "YOLOv8 Detection" and 'detections' in locals() and detections:
                            # Get detection classes
                            class_counts = {}
                            for d in detections:
                                cls = d["class"]
                                class_counts[cls] = class_counts.get(cls, 0) + 1
                            
                            # Create pie chart
                            fig, ax = plt.subplots(figsize=(8, 6))
                            
                            labels = list(class_counts.keys())
                            sizes = list(class_counts.values())
                            
                            # Use grayscale colors for black and white theme
                            colors = ['#000000', '#333333', '#666666', '#999999', '#BBBBBB', '#DDDDDD']
                            # Ensure enough colors for all labels
                            while len(colors) < len(labels):
                                colors.extend(colors)
                            colors = colors[:len(labels)]
                            
                            ax.pie(sizes, labels=None, autopct='%1.1f%%', startangle=90, colors=colors)
                            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                            
                            plt.title('Detection Distribution')
                            ax.set_facecolor('#FFFFFF')
                            fig.patch.set_facecolor('#FFFFFF')
                            
                            # Create legend
                            ax.legend(labels, loc="best", bbox_to_anchor=(1, 0.5))
                            
                            st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error processing image: {e}")
        else:
            st.warning("No images found in the specified dataset path.")
            
            # Show directory structure to help troubleshoot
            if os.path.exists(dataset_path):
                st.markdown('<div class="text">Directory contents:</div>', unsafe_allow_html=True)
                dir_contents = os.listdir(dataset_path)
                if dir_contents:
                    st.write(dir_contents[:10])  # Show only first 10 items
                    if len(dir_contents) > 10:
                        st.write(f"... and {len(dir_contents)-10} more items")
                else:
                    st.write("Directory is empty")

    else:  # Upload image option
        st.markdown('<div class="subtitle">Upload Your Own Image</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload a shelf image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Load and display the image
            image = Image.open(uploaded_file).convert("RGB")
            
            # Show original image and processed image side by side
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="text">Original Image</div>', unsafe_allow_html=True)
                st.image(image, use_container_width=True)
            
            with col2:
                st.markdown('<div class="text">Processed Image</div>', unsafe_allow_html=True)
                
                with st.spinner("Processing image..."):
                    if detection_method == "YOLOv8 Detection":
                        # Process with YOLO
                        processed_img, detections = process_with_yolo(image, model_path, confidence_threshold)
                        st.image(processed_img, use_container_width=True)
                        
                        # Show detection summary
                        if detections:
                            st.markdown(f'<div class="text">YOLOv8 detected {len(detections)} objects</div>', unsafe_allow_html=True)
                            
                            # Create a DataFrame for the detections
                            detection_df = pd.DataFrame([
                                {
                                    "Class": d["class"], 
                                    "Confidence": f"{d['confidence']*100:.1f}%",
                                    "Position": f"[{d['bbox'][0]}, {d['bbox'][1]}, {d['bbox'][2]}, {d['bbox'][3]}]"
                                }
                                for d in detections
                            ])
                            
                            # Display the detection table
                            st.dataframe(detection_df, use_container_width=True)
                            
                            # Summarize by class
                            class_counts = {}
                            for d in detections:
                                cls = d["class"]
                                class_counts[cls] = class_counts.get(cls, 0) + 1
                            
                            # Display summary
                            st.markdown('<div class="text">Detected Objects:</div>', unsafe_allow_html=True)
                            for cls, count in class_counts.items():
                                st.markdown(f'<div class="text">• {cls}: {count}</div>', unsafe_allow_html=True)
                        else:
                            st.warning("No objects detected in this image.")
                    else:
                        # Show image with contour detection
                        processed_img = show_contour_detection(image)
                        st.image(processed_img, use_column_width=True)
                        st.info("Basic contour detection is shown. This is not using the YOLO model.")
            
            # Image analysis and metrics
            if show_metrics:
                st.markdown('<div class="subtitle">Image Analysis</div>', unsafe_allow_html=True)
                
                with st.spinner("Analyzing image..."):
                    if detection_method == "YOLOv8 Detection" and 'detections' in locals():
                        metrics = analyze_image(image, detections)
                    else:
                        metrics = analyze_image(image)
                
                # Display metrics in a nice layout
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-label">BRIGHTNESS</div>
                        <div class="metric-value">{metrics['brightness']:.1f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-label">CONTRAST</div>
                        <div class="metric-value">{metrics['contrast']:.1f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-label">PRODUCTS</div>
                        <div class="metric-value">{metrics.get('product_count', 0)}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col4:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-label">PRICES</div>
                        <div class="metric-value">{metrics.get('price_count', 0)}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # If detections exist, show a pie chart of detections by class
                if detection_method == "YOLOv8 Detection" and 'detections' in locals() and detections:
                    # Get detection classes
                    class_counts = {}
                    for d in detections:
                        cls = d["class"]
                        class_counts[cls] = class_counts.get(cls, 0) + 1
                    
                    # Create pie chart
                    fig, ax = plt.subplots(figsize=(8, 6))
                    
                    labels = list(class_counts.keys())
                    sizes = list(class_counts.values())
                    
                    # Use grayscale colors for black and white theme
                    colors = ['#000000', '#333333', '#666666', '#999999', '#BBBBBB', '#DDDDDD']
                    # Ensure enough colors for all labels
                    while len(colors) < len(labels):
                        colors.extend(colors)
                    colors = colors[:len(labels)]
                    
                    ax.pie(sizes, labels=None, autopct='%1.1f%%', startangle=90, colors=colors)
                    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                    
                    plt.title('Detection Distribution')
                    ax.set_facecolor('#FFFFFF')
                    fig.patch.set_facecolor('#FFFFFF')
                    
                    # Create legend
                    ax.legend(labels, loc="best", bbox_to_anchor=(1, 0.5))
                    
                    st.pyplot(fig)

    # Add footer
    st.markdown("""
    <div style="text-align: center; margin-top: 30px; padding: 20px; 
               border-top: 3px solid #000000; font-family: 'Roboto Mono', monospace; background-color: #000000;">
        <div style="color: #FFFFFF; font-size: 1.2em; margin-bottom: 10px;">SUPERMARKET SHELF DETECTOR</div>
        <div style="color: #FFFFFF; font-size: 0.9em;">Created with YOLOv8 Custom-Trained Model</div>
    </div>
    """, unsafe_allow_html=True)

# Run the application
if __name__ == "__main__":
    main()
