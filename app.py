import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import streamlit as st
import torch
import gdown  # <-- added for Google Drive download

# Fix for OpenMP and Streamlit compatibility (if needed)
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Streamlit UI configuration
st.set_page_config(
    page_title="YOLO Object Detection",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 YOLO Object Detection")
st.markdown("Upload an image and detect objects using your YOLO model.")

# ----------------------------
# Google Drive Model Download
# ----------------------------
def download_model_from_drive(file_id: str, output_name: str = "yolo11n.pt"):
    """
    Downloads the model from Google Drive if it does not exist.
    """
    if not os.path.exists(output_name):
        st.info("Downloading YOLO model from Google Drive... Please wait.")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_name, quiet=False)
    return output_name

# Set your Google Drive file ID here
GOOGLE_DRIVE_FILE_ID = "YOUR_FILE_ID_HERE"
MODEL_FILE = download_model_from_drive(GOOGLE_DRIVE_FILE_ID)

# Load YOLO model once
@st.cache_resource
def load_model(model_path):
    try:
        # Check if it's an ONNX model or PyTorch model
        if model_path.endswith('.onnx'):
            st.info(f"Loading ONNX model: {model_path}")
            # YOLO can handle ONNX files directly when properly exported
            model = YOLO(model_path)
        else:
            st.info(f"Loading PyTorch model: {model_path}")
            model = YOLO(model_path)
        
        return model
    except Exception as e:
        st.error(f"⚠ Model loading failed: {e}")
        return None

def convert_to_onnx(model_path, output_path=None):
    try:
        if output_path is None:
            output_path = os.path.splitext(model_path)[0] + ".onnx"
            
        model = YOLO(model_path)
        success = model.export(format="onnx", imgsz=640)
        
        if success:
            return True, output_path
        else:
            return False, "Conversion failed"
    except Exception as e:
        return False, f"Error during conversion: {e}"

def process_image(image: Image.Image, model: YOLO, confidence: float = 0.25, 
                 class_filter=None, iou_threshold=0.45):
    img_array = np.array(image.convert("RGB"))

    # Run inference with improved parameters
    results = model(img_array, conf=confidence, iou=iou_threshold, 
                   augment=True)  # Enable augmentation for better detection
    result = results[0]
    
    # Create a copy for drawing
    annotated_img = img_array.copy()
    
    # Count detections by class
    detections_count = {}
    
    for box in result.boxes:
        conf_score = float(box.conf[0].cpu().numpy())
        class_id = int(box.cls[0].cpu().numpy())
        class_name = result.names.get(class_id, f"ID-{class_id}")
        
        # Apply class filtering if specified
        if class_filter and class_name not in class_filter:
            continue
            
        # Count detections
        detections_count[class_name] = detections_count.get(class_name, 0) + 1
        
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

        label = f"{class_name} {conf_score:.2f}"
        
        # Adjust colors based on class (person = red for better visibility)
        color = (0, 0, 255) if class_name == "person" else (0, 255, 0)
        
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
        
        # Improve text visibility with background
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(annotated_img, 
                     (x1, y1 - text_size[1] - 10), 
                     (x1 + text_size[0], y1), 
                     color, -1)
        cv2.putText(annotated_img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return annotated_img, detections_count


# Main Streamlit interface
def main():
    # Use session state to manage dynamic model addition
    if 'model_converted' not in st.session_state:
        st.session_state.model_converted = False
    if 'converted_model_path' not in st.session_state:
        st.session_state.converted_model_path = None
    
    # Model selection - initially only show the default model
    model_options = {"Custom Model (yolo11n.pt)": MODEL_FILE}  # <-- use downloaded model
    
    # Only scan for ONNX models when they exist
    onnx_models = [f for f in os.listdir() if f.endswith(".onnx") and os.path.isfile(f)]
    for model in onnx_models:
        model_name = os.path.splitext(model)[0]
        model_options[f"{model_name} (ONNX)"] = model
        
    # If a model was just converted, make it the selected model
    if st.session_state.model_converted and st.session_state.converted_model_path:
        if os.path.exists(st.session_state.converted_model_path):
            model_name = os.path.splitext(os.path.basename(st.session_state.converted_model_path))[0]
            model_options[f"{model_name} (ONNX)"] = st.session_state.converted_model_path
    
    # Sidebar for model options and conversion
    with st.sidebar:
        st.header("Model Settings")
        selected_model = st.selectbox(
            "Select Detection Model", 
            list(model_options.keys())
        )
        model_path = model_options[selected_model]
        
        # Check if model file exists except for the default one
        if model_path != MODEL_FILE and not os.path.exists(model_path):
            st.warning(f"Model {model_path} not found. Will download it automatically when used.")
        
        # ONNX conversion section
        st.header("ONNX Conversion")
        
        # Get available PT models for conversion
        available_pt_models = [MODEL_FILE] + [m for m in os.listdir() if m.endswith(".pt") and os.path.isfile(m) and m != MODEL_FILE]
        convert_model = st.selectbox(
            "Select Model to Convert", 
            available_pt_models
        )
        
        # Handle conversion button click
        if st.button("Convert to ONNX"):
            with st.spinner("Converting model to ONNX format..."):
                success, message = convert_to_onnx(convert_model)
                if success:
                    st.session_state.model_converted = True
                    st.session_state.converted_model_path = message
                    st.success(f"✅ Model successfully converted to {message}")
                    st.info("The converted model has been added to the selection menu.")
                else:
                    st.error(f"❌ {message}")
        
        # Advanced detection settings
        st.header("Detection Settings")
        confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)
        iou_threshold = st.slider("IoU Threshold", 0.1, 0.9, 0.45, 0.05, 
                                help="Lower values detect more overlapping objects")
        
        # Class filtering
        st.subheader("Class Filtering")
        filter_classes = st.checkbox("Filter by class", True)
        class_filter = None
        if filter_classes:
            class_options = ["person", "car", "truck", "bus", "motorcycle", "bicycle"]
            selected_classes = st.multiselect(
                "Classes to detect",
                class_options,
                default=["person", "car"]
            )
            if selected_classes:
                class_filter = selected_classes

    # Load the selected model
    model = load_model(model_path)
    if model is None:
        st.stop()

    # Main content area
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🖼 Original Image")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("🎯 Detected Objects")
            try:
                result_image, detections_count = process_image(
                    image, model, confidence, class_filter, iou_threshold
                )
                st.image(result_image, use_container_width=True)

                # Show detection counts
                if detections_count:
                    st.subheader("Detection Results")
                    for cls, count in detections_count.items():
                        st.metric(f"{cls}", count)

                # Convert image to downloadable format
                img_bytes = cv2.imencode(".jpg", cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))[1].tobytes()

                with st.expander("📥 Download Result Image"):
                    st.download_button(
                        label="Download Image",
                        data=img_bytes,
                        file_name="detection_result.jpg",
                        mime="image/jpeg"
                    )
            except Exception as e:
                st.error(f"Error during object detection: {e}")

if __name__ == "__main__":
    main()
