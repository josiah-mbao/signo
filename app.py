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
    # Y increases downwards. Tip must be "above" PIP (smaller Y value) to be open.
    tip_y = landmarks[tip_id].y
    pip_y = landmarks[pip_id].y
    return tip_y < pip_y

def classify_sign_geometric(landmarks):
    """
    Classifies a small subset of easily distinguishable signs using geometric rules,
    labeled as KSL handshapes for the purpose of the demo.
    """
    FINGERS = {
        'thumb': (4, 3), 'index': (8, 6), 'middle': (12, 10), 
        'ring': (16, 14), 'pinky': (20, 18)
    }
    
    fingers_open = {name: is_finger_open(landmarks, tip, pip) 
                    for name, (tip, pip) in FINGERS.items()}

    # --- Key Landmark Positions ---
    # Landmarks: [0:Wrist, 4:Thumb Tip, 8:Index Tip, 12:Middle Tip, 16:Ring Tip, 20:Pinky Tip]
    thumb_tip_x = landmarks[4].x
    index_mcp_x = landmarks[5].x # Metacarpophalangeal joint of index finger (base)

    
    # ✊ FIST / KSL Handshape 'S' (All fingers closed, thumb wraps over fingers)
    if all(not fingers_open[f] for f in ['index', 'middle', 'ring', 'pinky']):
        # Check if thumb is also mostly tucked (y position near wrist/base)
        if landmarks[4].y > landmarks[3].y:
            return "✊ FIST / KSL Handshape 'S'"
        
    # ✋ OPEN PALM / KSL Handshape 'B' (All fingers open, thumb tucked across)
    if all(fingers_open[f] for f in ['index', 'middle', 'ring', 'pinky']):
        # If thumb is closed (below PIP) AND thumb tip is to the right of the index MCP
        if not fingers_open['thumb'] and landmarks[4].x > landmarks[5].x:
            return "✋ OPEN PALM / KSL Handshape 'B'"

    # 🅰️ KSL Handshape 'A' (Fist, thumb on side)
    if all(not fingers_open[f] for f in ['index', 'middle', 'ring', 'pinky']):
         if fingers_open['thumb'] and thumb_tip_x < index_mcp_x:
            return "🅰️ KSL Handshape 'A'"
    
    # 🤟 KSL Handshape 'L' (Index and Thumb open, others closed)
    if fingers_open['index'] and fingers_open['thumb'] and not any([
        fingers_open['middle'], fingers_open['ring'], fingers_open['pinky']
    ]):
        return "🤟 KSL Handshape 'L'"

    # ✌️ KSL Handshape 'V' (Index and Middle open, others closed)
    if (fingers_open['index'] and fingers_open['middle'] and 
        not fingers_open['ring'] and not fingers_open['pinky']):
        return "✌️ KSL Handshape 'V'"

    # 🤙 KSL Handshape 'Y' (Thumb and Pinky extended)
    if fingers_open['thumb'] and fingers_open['pinky'] and not any([
        fingers_open['index'], fingers_open['middle'], fingers_open['ring']
    ]):
        return "🤙 KSL Handshape 'Y'"

    # 🖕 KSL Handshape 'I' (Pinky extended, others closed)
    if fingers_open['pinky'] and not any([
        fingers_open['thumb'], fingers_open['index'], fingers_open['middle'], fingers_open['ring']
    ]):
        return "🖕 KSL Handshape 'I'"
        
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
        # Handle empty/corrupted file gracefully
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
        return 0 # No data saved
    
    row = [label] + features
    with open(DATA_PATH, mode='a', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(row)
    return 1 # Data saved successfully

# --- 6. Sidebar UI and Mode Selection ---

current_samples = load_current_samples()

with col_sidebar:
    # --- NEW ABOUT SECTION ---
    st.markdown("---")
    st.header("About Signo: KSL Translator")
    st.markdown("""
    Signo is a proof-of-concept application designed to bridge communication gaps by recognizing hand gestures for **Kenya Sign Language (KSL)**.

    **Purpose:** To demonstrate the technical feasibility of real-time sign language translation using computer vision.

    **Tech Stack:**
    * **Frontend/UI:** Streamlit (Python web framework)
    * **Computer Vision:** MediaPipe Hands (Google's hand tracking solution) and OpenCV (video processing).
    * **Classification:** Geometric rules (for quick demo) or Machine Learning (for full KSL alphabet).

    **Demo Intent:** The "Quick Demo Mode" classifies common KSL handshapes based on geometric rules. The "Data Collection Mode" is the vital first step toward training a dedicated Machine Learning model for the full KSL alphabet.
    """)
    st.markdown("---")
    # --- End ABOUT SECTION ---
    
    st.header("1. Application Mode")
    
    app_mode = st.radio(
        "Select Operation Mode:",
        ("Quick Demo Mode (Geometric Rules)", "Data Collection Mode (ML Prep)"),
        index=0, # Default to Demo Mode
        help="Demo Mode shows instant classification of KSL handshapes. Data Collection Mode saves hand data to CSV."
    )
    
    st.markdown("---")
    
    if app_mode == "Quick Demo Mode (Geometric Rules)":
        st.header("Supported KSL Handshapes")
        st.markdown("""
            This demo uses simple geometry and is not a final ML model.
            - **✊ FIST / KSL 'S'**
            - **✋ OPEN PALM / KSL 'B'**
            - **🅰️ KSL Handshape 'A'**
            - **🤟 KSL Handshape 'L'**
            - **✌️ KSL Handshape 'V'**
            - **🤙 KSL Handshape 'Y'**
            - **🖕 KSL Handshape 'I'**
        """)
        status_info_text = st.info("Run the webcam to start real-time geometric classification.")
        
    else: # Data Collection Mode
        st.header("2. KSL Data Collection Setup")
        st.warning("Ensure your hand is clearly visible and centered.")
        
        selected_letter = st.selectbox(
            "Select KSL Sign to Record:",
            options=KSL_LETTERS
        )
        
        num_samples_to_record = st.slider(
            f"Target Samples for '{selected_letter}':",
            min_value=50, max_value=500, value=200, step=10
        )
        
        if 'recording' not in st.session_state:
            st.session_state.recording = False
        
        # Toggle button for recording
        if st.button("Toggle Recording"):
            st.session_state.recording = not st.session_state.recording
            st.experimental_rerun()
            
        recording_status = st.empty()
        
        # Display current progress
        st.markdown("---")
        st.subheader("Current Data Progress")
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
    # Use nested columns to center the video within the main column
    _, col_video, _ = st.columns([1, 6, 1])
    
    with col_video:
        run = st.checkbox("Start Webcam", value=False)
        FRAME_WINDOW = st.image([])
        # Dedicated placeholder for the gesture status text
        status_display = st.empty()

# --- 8. Main Webcam Loop ---

cap = None

if run:
    with st.spinner("Starting webcam..."):
        # Initialize video capture (using CAP_AVFOUNDATION for macOS compatibility)
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        
        if not cap.isOpened():
            st.error("Failed to access webcam. Check permissions and try restarting.")
            run = False
            
    while run and cap.isOpened():
        
        ret, frame = cap.read()
        if not ret:
            status_display.error("Lost webcam feed.")
            break

        # Convert the frame to RGB for MediaPipe processing
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1) # Flip the frame horizontally
        
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
                    mp_drawing.DrawingSpec(color=(240, 240, 240), thickness=2, circle_radius=2), # white
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2) # green
                )
                
                landmarks_list = hand_landmarks.landmark

                # --- MODE SWITCHING LOGIC ---
                if app_mode == "Data Collection Mode (ML Prep)":
                    
                    if st.session_state.recording:
                        current_count = current_samples.get(selected_letter, 0)
                        
                        if current_count < num_samples_to_record:
                            # 1. Extract features (63 landmark coords)
                            features = extract_landmarks(result.multi_hand_landmarks)
                            
                            # 2. Save data
                            saved_count = save_data(selected_letter, features)
                            
                            if saved_count:
                                current_samples[selected_letter] = current_count + saved_count
                            
                            gesture_text = f"RECORDING '{selected_letter}' - {current_samples.get(selected_letter, 0)}/{num_samples_to_record}"
                            update_progress_ui() 
                            
                            # Add green box around the image for visual feedback
                            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 15)
                            
                        else:
                            # Stop recording if target reached
                            st.session_state.recording = False
                            recording_status.success(f"Target of {num_samples_to_record} for '{selected_letter}' REACHED!")
                            st.experimental_rerun() 
                            
                    else:
                        gesture_text = f"Ready to record '{selected_letter}'."
                
                else: # Quick Demo Mode
                    # CLASSIFY using Geometric Rules
                    gesture_text = classify_sign_geometric(landmarks_list)


            # Display status
            if app_mode == "Data Collection Mode (ML Prep)" and st.session_state.recording:
                status_display.info(f"## **🔴 Recording: {gesture_text}**")
            else:
                status_display.success(f"## Recognized: **{gesture_text}**")

        else:
            status_display.info("## Recognized: *No hand detected*")

        # Display the processed frame
        FRAME_WINDOW.image(frame, channels="RGB")
        
        # Check the state of the webcam checkbox
        if not st.session_state.get('Start Webcam', True): 
             break
        
        # Prevent Streamlit from running too fast for the camera
        time.sleep(0.01)

# --- 9. Cleanup ---
if cap and cap.isOpened():
    cap.release()
    hands.close()
    
# Final status updates outside the loop
if not run and st.session_state.get('recording', False):
    st.session_state.recording = False
    
if not run:
    FRAME_WINDOW.empty()
    status_display.empty()
    status_display.info("Click 'Start Webcam' to begin.")
    
    if app_mode == "Data Collection Mode (ML Prep)":
        recording_status.empty()
    
if cap and not cap.isOpened():
    status_display.warning("Webcam stopped.")
