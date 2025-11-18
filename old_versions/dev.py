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

# Initialize theme state (defaulting to light to match Streamlit default)
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'
if 'last_toggle_time' not in st.session_state:
    st.session_state.last_toggle_time = 0

# --- 2. Theme CSS Injection ---
def apply_theme_css():
    """Injects CSS to simulate dynamic light/dark mode based on session state."""
    if st.session_state.theme == 'dark':
        css = """
        <style>
            /* Main Content and Background */
            [data-testid="stAppViewContainer"] {
                background-color: #0E1117; 
            }
            /* Sidebar Background */
            [data-testid="stSidebar"] {
                background-color: #1E232A; 
            }
            /* Text Color for all elements (headers, labels, markdown) */
            .stMarkdown, label, h1, h2, h3, h4, .st-bh, .st-bb {
                color: #FFFFFF;
            }
            /* Specific overrides for visual contrast */
            .stAlert {
                background-color: #333333 !important;
                color: #FFFFFF !important;
            }
            /* Overwrite Streamlit's default dark mode text color for primary components */
            .stSelectbox, .stSlider, .stRadio > label {
                color: #FFFFFF; 
            }
            /* Ensure core headers are white in dark mode */
            [data-testid="stTitle"], [data-testid="stHeader"] h1, [data-testid="stHeader"] h2, [data-testid="stHeader"] h3, [data-testid="stHeader"] h4 {
                 color: #FFFFFF !important;
            }
        </style>
        """
    else:
        # Soft Light Mode (Better contrast for dark text)
        css = """
        <style>
            /* Main Content and Background - Soft light blue/gray for better contrast */
            [data-testid="stAppViewContainer"] {
                background-color: #F7F9FF; 
            }
            /* Sidebar Background - Slightly darker soft color */
            [data-testid="stSidebar"] {
                background-color: #EAEFF7;
            }
            /* Explicitly set all standard text and form elements to dark color */
            .stMarkdown, label, [data-testid="stMetric"], 
            .stSelectbox, .stSlider, .stRadio > label,
            
            /* FIX: Target Streamlit Headings with data-testid and h tags + !important */
            h1, h2, h3, h4, 
            [data-testid="stTitle"], 
            [data-testid="stHeader"],
            [data-testid="stSidebarHeader"] {
                color: #262730 !important; /* Dark Streamlit Text color for max readability */
            }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)

# Apply CSS immediately after set_page_config (for initial load)
apply_theme_css()

# --- 3. Streamlit Configuration and Layout ---
st.set_page_config(page_title="Signo - KSL Data Collection & Demo", page_icon="✋", layout="wide")

st.title("🇰🇪 Signo: Kenya Sign Language (KSL) Gesture Translator Demo")

col_main, col_sidebar = st.columns([4, 1.5]) 

# --- 4. Mediapipe Setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1, # Focus on one hand for single-letter KSL
    min_detection_confidence=0.8, 
    min_tracking_confidence=0.6
)

# --- 5. Geometric Classification Logic (for quick Demo Mode) ---

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

    # Key Landmarks
    thumb_tip_x = landmarks[4].x
    index_mcp_x = landmarks[5].x 
    # Check if thumb tip (4) is higher (smaller Y) than the index PIP (6)
    thumb_is_high = landmarks[4].y < landmarks[6].y
    
    # --- New: THUMBS UP / TOGGLE GESTURE ---
    if thumb_is_high and all(not fingers_open[f] for f in ['index', 'middle', 'ring', 'pinky']):
        return "👍 THUMBS UP (Mode Toggle)"
    
    # --- Existing KSL Signs ---
    
    # ✊ FIST / KSL Handshape 'S' (All fingers closed, thumb wraps over fingers)
    if all(not fingers_open[f] for f in ['index', 'middle', 'ring', 'pinky']):
        if landmarks[4].y > landmarks[3].y:
            return "✊ FIST / KSL 'S'"
            
    # ✋ OPEN PALM / KSL Handshape 'B' (All fingers open, thumb tucked across)
    if all(fingers_open[f] for f in ['index', 'middle', 'ring', 'pinky']):
        if not fingers_open['thumb'] and landmarks[4].x > landmarks[5].x:
            return "✋ OPEN PALM / KSL 'B'"

    # 🅰️ KSL Handshape 'A' (Fist, thumb on side)
    if all(not fingers_open[f] for f in ['index', 'middle', 'ring', 'pinky']):
         if fingers_open['thumb'] and thumb_tip_x < index_mcp_x:
            return "🅰️ KSL 'A'"
    
    # 🤟 KSL Handshape 'L' (Index and Thumb open, others closed)
    if fingers_open['index'] and fingers_open['thumb'] and not any([
        fingers_open['middle'], fingers_open['ring'], fingers_open['pinky']
    ]):
        return "🤟 KSL 'L'"

    # ✌️ KSL Handshape 'V' (Index and Middle open, others closed)
    if (fingers_open['index'] and fingers_open['middle'] and 
        not fingers_open['ring'] and not fingers_open['pinky']):
        return "✌️ KSL 'V'"

    # 🤙 KSL Handshape 'Y' (Thumb and Pinky extended)
    if fingers_open['thumb'] and fingers_open['pinky'] and not any([
        fingers_open['index'], fingers_open['middle'], fingers_open['ring']
    ]):
        return "🤙 KSL 'Y'"

    # 🖕 KSL Handshape 'I' (Pinky extended, others closed)
    if fingers_open['pinky'] and not any([
        fingers_open['thumb'], fingers_open['index'], fingers_open['middle'], fingers_open['ring']
    ]):
        return "🖕 KSL 'I'"
        
    return "🤷 Unknown"


# --- 6. Helper Functions (Data Collection) ---

@st.cache_data
def load_current_samples():
    """Load existing data to show current sample counts."""
    if not os.path.exists(DATA_PATH):
        return {}
    
    try:
        # UserWarning: loadtxt: input contained no data: "KSL_dataset.csv" is handled by the try/except block
        data = np.loadtxt(DATA_PATH, delimiter=',', skiprows=1, dtype=str)
        if data.ndim == 1 and data.size > 0:
            data = data.reshape(1, -1)
            
        if data.size == 0:
            return {}
        
        labels = data[:, 0]
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique, counts))
    except Exception:
        # Handle case where file exists but is empty/malformed
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

# --- 7. Sidebar UI and Mode Selection ---

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
        st.info("The video border will be **Blue** in this mode. Use the **👍 THUMBS UP** gesture to toggle the theme!")
        
        # Supported signs details inside a small expander
        with st.expander("View Supported KSL Handshapes"):
            st.markdown("""
            This demo uses simple geometry:
            - **👍 THUMBS UP (Mode Toggle)**
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
        st.warning("Ensure your hand is clearly visible and centered. Border will be **Red** when recording.")
        
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
            # st.rerun() removed
            
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

# --- 8. Main Video Feed Setup ---

with col_main:
    st.subheader("Real-Time KSL Gesture Recognition")
    
    run = st.checkbox("Start Webcam Feed", value=False)
    FRAME_WINDOW = st.image([])
    
# --- 9. Main Webcam Loop ---

cap = None
prev_time = 0 

if run:
    with st.spinner("Starting webcam..."):
        # Note: If running on Mac/Linux, you might need to adjust the video capture index 
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
        
        # Call apply_theme_css() here to update the style dynamically without rerun
        apply_theme_css() 

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
                            # st.rerun() removed
                            
                    else:
                        gesture_text = f"Ready to record: '{selected_letter}'"
                
                else: # Quick Demo Mode
                    gesture_text = classify_sign_geometric(landmarks_list)
                    
                    # --- UI TOGGLE LOGIC ---
                    if gesture_text == "👍 THUMBS UP (Mode Toggle)":
                        current_time = time.time()
                        if current_time - st.session_state.last_toggle_time > 1.5:
                            st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
                            st.session_state.last_toggle_time = current_time
                            # Optional Feedback
                            st.info(f"Switched to {st.session_state.theme.capitalize()} Mode", icon="✨")


        
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
        
        # 1. Determine text colors 
        
        # Gesture text overlay is on the video feed, so we typically keep it white
        # or choose based on the video background. White works well against the 
        # semi-transparent black box we draw.
        gesture_text_color_bgr = (255, 255, 255) 

        # 2. FPS counter color is set to bright green (0, 255, 0) (FIX 2)
        fps_text_color_bgr = (0, 255, 0) 
        
        # Draw FPS counter (Top Left - Green Text)
        cv2.putText(frame, f'FPS: {int(fps)}', (25, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, fps_text_color_bgr, 2, cv2.LINE_AA)

        # Draw Gesture Text Overlay (Centered on Frame, bottom)
        text_color_bgr = gesture_text_color_bgr 
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
        # Check if the webcam checkbox has been unticked by the user
        if not run: 
             break
        
        time.sleep(0.01)

# --- 10. Cleanup and Final Status ---
if cap and cap.isOpened():
    cap.release()
    hands.close()
    
# Final status updates outside the loop
if not run and st.session_state.get('recording', False):
    st.session_state.recording = False
    
if not run:
    FRAME_WINDOW.empty()
    
    # Re-displaying simple start message in the subheader area
    with col_main:
        st.subheader("Real-Time KSL Gesture Recognition")
        st.info("Click 'Start Webcam Feed' to begin.")
    
    if app_mode == "Data Collection Mode (ML Prep)":
        if 'recording_status' in locals():
            recording_status.empty()
    
if cap and not cap.isOpened():
    st.warning("Webcam stopped.")
