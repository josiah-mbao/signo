import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import os
import csv
import time

# --- 1. Constants and Setup ---
# The path where the dataset will be saved
DATA_PATH = 'KSL_dataset.csv' 
# Using a general alphabet list, but contextualizing it for KSL data collection
KSL_LETTERS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'SPACE', 'DELETE'
]
NUM_FEATURES = 63 

# Initialize CSV file if it doesn't exist (updated filepath to KSL_dataset.csv)
if not os.path.exists(DATA_PATH):
    header = ['label'] + [f'landmark_{i}_{coord}' for i in range(21) for coord in ['x', 'y', 'z']]
    with open(DATA_PATH, mode='w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(header)

# --- 2. Streamlit Configuration and Layout ---
st.set_page_config(page_title="Signo - KSL Data Collection & Demo", page_icon="✋", layout="wide")

st.title("🇰🇪 Signo: Kenya Sign Language (KSL) Gesture Translator Demo")

col_main, col_sidebar = st.columns([4, 1.5]) 

# --- 3. Mediapipe Setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1, # Focus on one hand for single-letter KSL
    min_detection_confidence=0.8, 
    min_tracking_confidence=0.6
)

# --- 4. Geometric Classification Logic (for quick Demo Mode) ---

def is_finger_open(landmarks, tip_id, pip_id):
    """Checks if a finger is open by comparing y-coordinate of tip vs PIP joint."""
    tip_y = landmarks[tip_id].y
    pip_y = landmarks[pip_id].y
    return tip_y < pip_y

def classify_sign_geometric(landmarks):
    """Classifies a small subset of KSL handshapes using geometric rules."""
    FINGERS = {
        'thumb': (4, 3), 'index': (8, 6), 'middle': (12, 10), 
        'ring': (16, 14), 'pinky': (20, 18)
    }
    
    fingers_open = {name: is_finger_open(landmarks, tip, pip) 
                    for name, (tip, pip) in FINGERS.items()}

    thumb_tip_x = landmarks[4].x
    index_mcp_x = landmarks[5].x 

    if all(not fingers_open[f] for f in ['index', 'middle', 'ring', 'pinky']):
        if landmarks[4].y > landmarks[3].y:
            return "✊ FIST / KSL 'S'"
    if all(fingers_open[f] for f in ['index', 'middle', 'ring', 'pinky']):
        if not fingers_open['thumb'] and landmarks[4].x > landmarks[5].x:
            return "✋ OPEN PALM / KSL 'B'"
    if all(not fingers_open[f] for f in ['index', 'middle', 'ring', 'pinky']):
         if fingers_open['thumb'] and thumb_tip_x < index_mcp_x:
            return "🅰️ KSL 'A'"
    if fingers_open['index'] and fingers_open['thumb'] and not any([
        fingers_open['middle'], fingers_open['ring'], fingers_open['pinky']
    ]):
        return "🤟 KSL 'L'"
    if (fingers_open['index'] and fingers_open['middle'] and 
        not fingers_open['ring'] and not fingers_open['pinky']):
        return "✌️ KSL 'V'"
    if fingers_open['thumb'] and fingers_open['pinky'] and not any([
        fingers_open['index'], fingers_open['middle'], fingers_open['ring']
    ]):
        return "🤙 KSL 'Y'"
    if fingers_open['pinky'] and not any([
        fingers_open['thumb'], fingers_open['index'], fingers_open['middle'], fingers_open['ring']
    ]):
        return "🖕 KSL 'I'"
        
    return "🤷 Unknown"


# --- 5. Helper Functions (Data Collection) ---

@st.cache_data
def load_current_samples():
    """Load existing data to show current sample counts."""
    if not os.path.exists(DATA_PATH):
        return {}
    
    try:
        data = np.loadtxt(DATA_PATH, delimiter=',', skiprows=1, dtype=str)
        if data.ndim == 1 and data.size > 0:
            data = data.reshape(1, -1)
            
        if data.size == 0:
            return {}
        
        labels = data[:, 0]
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique, counts))
    except Exception:
        return {}


def extract_landmarks(hand_landmarks):
    """Flattens the 21 landmarks (x, y, z) into a 63-element list."""
    if not hand_landmarks:
        return None
    
    landmarks = hand_landmarks[0].landmark 
    
    data = []
    for landmark in landmarks:
        data.extend([landmark.x, landmark.y, landmark.z])
    return data

def save_data(label, features):
    """Appends the label and features to the CSV file."""
    if features is None:
        return 0 
    
    row = [label] + features
    with open(DATA_PATH, mode='a', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(row)
    return 1

# --- 6. Sidebar UI and Mode Selection ---

current_samples = load_current_samples()
total_samples_collected = sum(current_samples.values()) 

with col_sidebar:
    
    # --- About Signo Expander ---
    with st.expander("About Signo and Tech Stack", expanded=False):
        st.markdown("""
        Signo is a proof-of-concept application designed to bridge communication gaps by recognizing hand gestures for **Kenya Sign Language (KSL)**.

        **Purpose:** To demonstrate the technical feasibility of real-time sign language translation using computer vision.

        **Tech Stack:**
        * **Frontend/UI:** Streamlit (Python web framework)
        * **Computer Vision:** MediaPipe Hands (Google's hand tracking solution) and OpenCV (video processing).
        * **Classification:** Geometric rules (for quick demo) or Machine Learning (for full KSL alphabet).
        """)
    st.markdown("---")
    
    # --- Samples Metric ---
    st.metric(label="Total Data Samples Collected", value=f"{total_samples_collected:,}")
    st.markdown("---")

    # --- Mode Selection and Controls Group ---
    st.header("1. Application Controls")
    
    app_mode = st.radio(
        "Select Operation Mode:",
        ("Quick Demo Mode (Geometric Rules)", "Data Collection Mode (ML Prep)"),
        index=0, 
        help="Demo Mode shows instant classification. Data Collection Mode saves hand data."
    )
    
    st.markdown("---")
    
    if app_mode == "Quick Demo Mode (Geometric Rules)":
        
        st.subheader("Quick Demo Setup")
        st.info("The video border will be **Blue** in this mode.")
        
        # Supported signs details inside a small expander
        with st.expander("View Supported KSL Handshapes"):
            st.markdown("""
            This demo uses simple geometry:
            - **✊ FIST / KSL 'S'**
            - **✋ OPEN PALM / KSL 'B'**
            - **🅰️ KSL Handshape 'A'**
            - **🤟 KSL Handshape 'L'**
            - **✌️ KSL Handshape 'V'**
            - **🤙 KSL Handshape 'Y'**
            - **🖕 KSL Handshape 'I'**
            """)
        
    else: # Data Collection Mode
        
        st.subheader("Data Collection Setup")
        st.warning("Ensure your hand is clearly visible and centered.")
        
        selected_letter = st.selectbox(
            "Select KSL Sign to Record:",
            options=KSL_LETTERS
        )
        
        num_samples_to_record = st.slider(
            f"Target Samples for '{selected_letter}':",
            min_value=50, max_value=500, value=200, step=10,
            help=f"Target samples required to train the ML model for '{selected_letter}'."
        )
        
        if 'recording' not in st.session_state:
            st.session_state.recording = False
        
        # Toggle button for recording
        button_label = "Stop Recording" if st.session_state.recording else "Start Recording"
        if st.button(button_label, use_container_width=True):
            st.session_state.recording = not st.session_state.recording
            st.experimental_rerun() # Rerun to update state/UI instantly
            
        recording_status = st.empty()
        st.markdown("---")
        
        # Display current progress
        st.subheader("Data Progress Tracker")
        progress_container = st.container(border=True)
        
        def update_progress_ui():
            """Helper to redraw progress indicators."""
            progress_container.empty()
            with progress_container:
                for letter in KSL_LETTERS:
                    count = current_samples.get(letter, 0)
                    st.progress(min(count / num_samples_to_record, 1.0), text=f"**{letter}** ({count} samples)")

        update_progress_ui()

# --- 7. Main Video Feed Setup ---

with col_main:
    # Centered video display with a dynamic border style
    st.subheader("Real-Time KSL Gesture Recognition")
    
    # Placeholder for the dynamic border style (CSS injection)
    border_style_css = st.empty()
    
    run = st.checkbox("Start Webcam Feed", value=False)
    FRAME_WINDOW = st.image([])
    
# --- 8. Main Webcam Loop ---

cap = None
prev_time = 0 

if run:
    with st.spinner("Starting webcam..."):
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        
        if not cap.isOpened():
            st.error("Failed to access webcam. Check permissions and try restarting.")
            run = False
            
    while run and cap.isOpened():
        
        ret, frame = cap.read()
        if not ret:
            st.error("Lost webcam feed.")
            break

        current_time = time.time()
        fps = 1 / (current_time - prev_time) if current_time != prev_time else 0
        prev_time = current_time

        # Convert the frame to RGB for MediaPipe processing
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1) 
        
        frame.flags.writeable = False
        result = hands.process(frame)
        frame.flags.writeable = True

        gesture_text = "No hand detected"
        
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(240, 240, 240), thickness=2, circle_radius=2), 
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2) 
                )
                
                landmarks_list = hand_landmarks.landmark

                # --- MODE SWITCHING LOGIC ---
                if app_mode == "Data Collection Mode (ML Prep)":
                    
                    if st.session_state.recording:
                        current_count = current_samples.get(selected_letter, 0)
                        
                        if current_count < num_samples_to_record:
                            features = extract_landmarks(result.multi_hand_landmarks)
                            
                            saved_count = save_data(selected_letter, features)
                            
                            if saved_count:
                                # Reload samples and update total count 
                                current_samples = load_current_samples()
                                
                            gesture_text = f"Recording: '{selected_letter}' ({current_samples.get(selected_letter, 0)}/{num_samples_to_record})"
                            update_progress_ui() 
                            
                        else:
                            st.session_state.recording = False
                            # Rerunning will trigger success message in sidebar 
                            st.experimental_rerun() 
                            
                    else:
                        gesture_text = f"Ready to record: '{selected_letter}'"
                
                else: # Quick Demo Mode
                    gesture_text = classify_sign_geometric(landmarks_list)

        
        # --- Determine Border Color (BGR) ---
        if app_mode == "Quick Demo Mode (Geometric Rules)":
            border_bgr = (255, 100, 0)  # Blue/Cyan for Demo
        elif app_mode == "Data Collection Mode (ML Prep)" and st.session_state.recording:
            border_bgr = (0, 0, 255)  # Red for Recording
        else: # Data Collection Idle / Hand not detected in Demo
            border_bgr = (0, 255, 0)  # Green for Ready / Idle

        # --- Draw Dynamic Border (Thick line on all sides) ---
        border_thickness = 15
        H, W, _ = frame.shape
        cv2.rectangle(frame, (0, 0), (W, H), border_bgr, border_thickness)


        # --- Draw Status and Metrics on Frame ---
        
        # Draw FPS counter (Top Left - White Text)
        cv2.putText(frame, f'FPS: {int(fps)}', (25, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Draw Gesture Text Overlay (Centered on Frame, bottom)
        text_color_bgr = (255, 255, 255) # White text
        text_scale = 1.5
        text_thickness = 3
        text_size = cv2.getTextSize(gesture_text, cv2.FONT_HERSHEY_DUPLEX, text_scale, text_thickness)[0]
        text_x = int((W - text_size[0]) / 2)
        text_y = int(H - 45) # Near bottom center
        
        # Draw background box for better contrast (Semi-transparent black background)
        cv2.rectangle(frame, (text_x - 10, text_y - text_size[1] - 10), (text_x + text_size[0] + 10, text_y + 10), (0, 0, 0), cv2.FILLED)
        
        # Draw the main gesture text
        cv2.putText(frame, gesture_text, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, text_scale, text_color_bgr, text_thickness, cv2.LINE_AA)

        # Display the processed frame
        FRAME_WINDOW.image(frame, channels="RGB")
        
        # Check the state of the webcam checkbox
        if not st.session_state.get('Start Webcam', True): 
             break
        
        time.sleep(0.01)

# --- 9. Cleanup and Final Status ---
if cap and cap.isOpened():
    cap.release()
    hands.close()
    
# Final status updates outside the loop
if not run and st.session_state.get('recording', False):
    st.session_state.recording = False
    
if not run:
    FRAME_WINDOW.empty()
    
    # Hide the FRAME_WINDOW placeholder until re-run
    # Re-displaying simple start message in the subheader area
    with col_main:
        st.subheader("Real-Time KSL Gesture Recognition")
        st.info("Click 'Start Webcam Feed' to begin.")
    
    if app_mode == "Data Collection Mode (ML Prep)":
        if 'recording_status' in locals():
            recording_status.empty()
    
if cap and not cap.isOpened():
    st.warning("Webcam stopped.")
