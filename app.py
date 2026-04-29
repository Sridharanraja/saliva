import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# --- Configuration ---
MODEL_PATH = "./weight/best_P_and_NP_V2_600.pt" 

@st.cache_resource
def load_model():
    """Loads the YOLO model only once to save resources."""
    try:
        model = YOLO(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def determine_status(results):
    """
    Determines pregnancy status based on detected classes:
    - FEL (Fern-like)   -> Non-Pregnant
    - BL (Branch-like)  -> Pregnant
    - FIL (Fir-like)    -> Pregnant
    """
    detected_class_names = []
    
    for r in results:
        if r.boxes:
            for class_id in r.boxes.cls.cpu().numpy():
                detected_class_names.append(r.names[int(class_id)])
    
    if 'BL' in detected_class_names or 'FIL' in detected_class_names:
        return "Pregnant"
    elif 'FEL' in detected_class_names:
        return "Non-Pregnant"
        
    return "No Pattern Detected"

def get_available_cameras():
    """Checks for available USB cameras and returns a list of indices."""
    available_cams = []
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cams.append(i)
            cap.release()
    return available_cams

# --- Main App Interface ---
st.set_page_config(page_title="Pregnancy Status Analyzer", layout="wide")
st.title("Cow Pregnancy Status Analyzer")
st.markdown("Upload a photo, capture an image, or use live video to detect patterns and determine status.")

model = load_model()

if model is None:
    st.stop()

# --- Global Settings (Home Page) ---
st.markdown("### Settings")
# The confidence slider determines how strict the model is before drawing a box
conf_threshold = st.slider(
    "Model Confidence Threshold", 
    min_value=0.10, 
    max_value=1.00, 
    value=0.50, 
    step=0.05,
    help="Increase this value if you see false positives. Decrease if the model is missing patterns."
)

st.divider()

# Create tabs for the different input methods
tab1, tab2, tab3 = st.tabs(["Upload Photo", "Capture Photo", "Live Video Stream"])

# --- TAB 1: Upload Photo ---
with tab1:
    st.header("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("Analyze Uploaded Image"):
            with st.spinner('Analyzing...'):
                # Pass the global confidence threshold here
                results = model.predict(source=image, conf=conf_threshold)
                annotated_image = results[0].plot()
                status = determine_status(results)
                
                st.subheader("Analysis Results")
                if status == "Pregnant":
                    st.success(f"Status: **{status}**")
                elif status == "Non-Pregnant":
                    st.info(f"Status: **{status}**")
                else:
                    st.warning(f"Status: **{status}**")
                    
                st.image(annotated_image, caption="Predicted Image with Segmentation", use_container_width=True)

# --- TAB 2: Capture Photo ---
with tab2:
    st.header("Capture from Webcam")
    
    # Checkbox controls whether the camera renders
    enable_camera = st.checkbox("Turn on Webcam")
    
    if enable_camera:
        camera_photo = st.camera_input("Take a picture")
        
        if camera_photo is not None:
            image = Image.open(camera_photo)
            
            if st.button("Analyze Captured Image"):
                with st.spinner('Analyzing...'):
                    # Pass the global confidence threshold here
                    results = model.predict(source=image, conf=conf_threshold)
                    annotated_image = results[0].plot()
                    status = determine_status(results)
                    
                    st.subheader("Analysis Results")
                    if status == "Pregnant":
                        st.success(f"Status: **{status}**")
                    elif status == "Non-Pregnant":
                        st.info(f"Status: **{status}**")
                    else:
                        st.warning(f"Status: **{status}**")
                        
                    st.image(annotated_image, caption="Predicted Image with Segmentation", use_container_width=True)
    else:
        st.write("Check the box above to activate your camera.")

# --- TAB 3: Live Video Stream ---
with tab3:
    st.header("Live Video Analysis")
    
    available_cameras = get_available_cameras()
    
    if not available_cameras:
        st.warning("No USB cameras detected. Please ensure your camera is connected.")
    else:
        selected_cam = st.selectbox("Select Camera Index", available_cameras)
        
        run_video = st.checkbox("Start Live Stream")
        FRAME_WINDOW = st.empty()
        STATUS_WINDOW = st.empty()
        
        if run_video:
            cap = cv2.VideoCapture(selected_cam)
            
            while run_video:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to grab frame from camera.")
                    break
                
                # Convert BGR to RGB for Streamlit/YOLO processing
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Pass the global confidence threshold here
                results = model.predict(source=frame_rgb, conf=conf_threshold, verbose=False)
                annotated_frame = results[0].plot()
                status = determine_status(results)
                
                # Display status dynamically
                if status == "Pregnant":
                    STATUS_WINDOW.success(f"Live Status: **{status}**")
                elif status == "Non-Pregnant":
                    STATUS_WINDOW.info(f"Live Status: **{status}**")
                else:
                    STATUS_WINDOW.warning(f"Live Status: **{status}**")
                
                # Update the image placeholder with the annotated frame
                FRAME_WINDOW.image(annotated_frame, channels="RGB", use_container_width=True)
            
            cap.release()
        else:
            st.write("Check the box above to start the live video stream.")
