import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import os
import csv
import time

# --- 1. Constants and Setup ---
DATA_PATH = 'KSL_dataset.csv'
KSL_LETTERS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'SPACE', 'DELETE'
]
NUM_FEATURES = 63

# Initialize CSV file if it doesn't exist
if not os.path.exists(DATA_PATH):
    header = ['label'] + [f'landmark_{i}_{coord}' for i in range(21) for coord in ['x', 'y', 'z']]
    with open(DATA_PATH, mode='w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(header)

# Initialize session state
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'
if 'last_toggle_time' not in st.session_state:
    st.session_state.last_toggle_time = 0
if 'feedback_display_time' not in st.session_state:
    st.session_state.feedback_display_time = 0
    
# Sentence building session state
if 'current_sentence' not in st.session_state:
    st.session_state.current_sentence = ""
if 'sentence_history' not in st.session_state:
    st.session_state.sentence_history = []
if 'last_phrase_time' not in st.session_state:
    st.session_state.last_phrase_time = 0
if 'gesture_history' not in st.session_state:
    st.session_state.gesture_history = []

# --- 2. Theme CSS Injection ---
def apply_theme_css():
    """Injects CSS to simulate dynamic light/dark mode based on session state."""
    if st.session_state.theme == 'dark':
        css = """
        <style>
            [data-testid="stAppViewContainer"] {
                background-color: #0E1117; 
            }
            [data-testid="stSidebar"] {
                background-color: #1E232A; 
            }
            .stMarkdown, label, h1, h2, h3, h4, .st-bh, .st-bb {
                color: #FFFFFF;
            }
            .stAlert {
                background-color: #333333 !important;
                color: #FFFFFF !important;
            }
            .stSelectbox, .stSlider, .stRadio > label {
                color: #FFFFFF; 
            }
            [data-testid="stTitle"], [data-testid="stHeader"] h1, [data-testid="stHeader"] h2, [data-testid="stHeader"] h3, [data-testid="stHeader"] h4 {
                 color: #FFFFFF !important;
            }
        </style>
        """
    else:
        css = """
        <style>
            [data-testid="stAppViewContainer"] {
                background-color: #FAF8F1;
            }
            [data-testid="stSidebar"] {
                background-color: #F0EEE9;
            }
            .stMarkdown, label, [data-testid="stMetric"], 
            .stSelectbox, .stSlider, .stRadio > label,
            h1, h2, h3, h4, 
            [data-testid="stTitle"], 
            [data-testid="stHeader"],
            [data-testid="stSidebarHeader"] {
                color: #262730 !important;
            }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)

apply_theme_css()

# --- 3. Streamlit Configuration and Layout ---
st.set_page_config(page_title="Signo - KSL Sentence Builder", page_icon="✋", layout="wide")
st.title("🇰🇪 Signo: KSL Sentence & Phrase Builder")

col_main, col_sidebar = st.columns([4, 1.5])

# --- 4. Mediapipe Setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.6
)

# --- 5. Enhanced Geometric Classification with Word Building Gestures ---

def is_finger_open(landmarks, tip_id, pip_id):
    """Checks if a finger is open by comparing y-coordinate of tip vs PIP joint."""
    tip_y = landmarks[tip_id].y
    pip_y = landmarks[pip_id].y
    return tip_y < pip_y

def detect_flat_hand(landmarks):
    """Detects if all fingers are extended (flat hand)."""
    FINGERS = {
        'thumb': (4, 3), 'index': (8, 6), 'middle': (12, 10), 
        'ring': (16, 14), 'pinky': (20, 18)
    }
    fingers_open = {name: is_finger_open(landmarks, tip, pip) 
                    for name, (tip, pip) in FINGERS.items()}
    
    return all(fingers_open[f] for f in ['index', 'middle', 'ring', 'pinky'])

def detect_fist(landmarks):
    """Detects if all fingers are closed (fist)."""
    FINGERS = {
        'index': (8, 6), 'middle': (12, 10), 'ring': (16, 14), 'pinky': (20, 18)
    }
    fingers_open = {name: is_finger_open(landmarks, tip, pip) 
                    for name, (tip, pip) in FINGERS.items()}
    
    return all(not fingers_open[f] for f in ['index', 'middle', 'ring', 'pinky'])

def detect_pinch(landmarks, tip1, tip2, threshold=0.05):
    """Detects if two fingertips are close together (pinch gesture)."""
    dist = np.sqrt(
        (landmarks[tip1].x - landmarks[tip2].x)**2 +
        (landmarks[tip1].y - landmarks[tip2].y)**2 +
        (landmarks[tip1].z - landmarks[tip2].z)**2
    )
    return dist < threshold

def classify_sign_geometric(landmarks):
    """Classifies handshapes including word-building gestures."""
    FINGERS = {
        'thumb': (4, 3), 'index': (8, 6), 'middle': (12, 10), 
        'ring': (16, 14), 'pinky': (20, 18)
    }
    
    fingers_open = {name: is_finger_open(landmarks, tip, pip) 
                    for name, (tip, pip) in FINGERS.items()}

    thumb_tip_x = landmarks[4].x
    index_mcp_x = landmarks[5].x
    thumb_is_high = landmarks[4].y < landmarks[6].y
    
    # --- SPACE GESTURE: Flat hand with thumb extended (like "B" but more relaxed)
    if (detect_flat_hand(landmarks) and 
        fingers_open['thumb'] and 
        landmarks[4].x > landmarks[5].x):
        return "␣ SPACE"
    
    # --- DELETE GESTURE: Pinch between thumb and index, then swipe motion detection
    if (detect_pinch(landmarks, 4, 8) and 
        all(not fingers_open[f] for f in ['middle', 'ring', 'pinky'])):
        return "⌫ DELETE"
    
    # --- ENTER/CONFIRM GESTURE: Thumbs up
    if thumb_is_high and all(not fingers_open[f] for f in ['index', 'middle', 'ring', 'pinky']):
        return "↵ ENTER"
    
    # --- Existing KSL Signs ---
    if detect_fist(landmarks):
        if landmarks[4].y > landmarks[3].y:
            return "✊ S"

    if (detect_flat_hand(landmarks) and
        not fingers_open['thumb'] and landmarks[4].x > landmarks[5].x):
        return "✋ B"

    if detect_fist(landmarks):
        if fingers_open['thumb'] and thumb_tip_x < index_mcp_x:
            return "🅰️ A"

    if (fingers_open['index'] and fingers_open['thumb'] and
        not any([fingers_open['middle'], fingers_open['ring'], fingers_open['pinky']])):
        return "🤟 L"

    if (fingers_open['index'] and fingers_open['middle'] and
        not fingers_open['ring'] and not fingers_open['pinky']):
        return "✌️ V"

    if (fingers_open['thumb'] and fingers_open['pinky'] and
        not any([fingers_open['index'], fingers_open['middle'], fingers_open['ring']])):
        return "🤙 Y"

    if (fingers_open['pinky'] and
        not any([fingers_open['thumb'], fingers_open['index'], fingers_open['middle'], fingers_open['ring']])):
        return "🖕 I"

    # --- Phrase Signs ---
    # HELLO: Flat hand with thumb on left (not B)
    if (detect_flat_hand(landmarks) and
        fingers_open['thumb'] and landmarks[4].x < landmarks[5].x):
        return "HELLO"

    # THANK YOU: Fist with thumb on right and extended
    if all(not fingers_open[f] for f in ['index', 'middle', 'ring', 'pinky']) and fingers_open['thumb'] and thumb_tip_x > index_mcp_x:
        return "THANK YOU"

    # GOOD DAY: V shape with thumb
    if (fingers_open['index'] and fingers_open['middle'] and fingers_open['thumb'] and
        not fingers_open['ring'] and not fingers_open['pinky']):
        return "GOOD DAY"

    return "🤷 Unknown"

# --- 6. Sentence Building Logic ---

def handle_sentence_building(gesture_text, current_time):
    """Process gestures for sentence building with timing constraints."""

    # Gesture to phrase mapping (mapped to demo-friendly words/phrases)
    gesture_to_phrase = {
        "HELLO": "HELLO",
        "THANK YOU": "THANK YOU",
        "GOOD DAY": "GOOD DAY",
        "✊ S": "I",
        "✋ B": "YOU",
        "🅰️ A": "AM",
        "🤟 L": "GOOD",
        "✌️ V": "WE",
        "🤙 Y": "SEE",
        "🖕 I": "ME"
    }

    # Process phrase gestures (with anti-spam delay)
    if gesture_text in gesture_to_phrase:
        if current_time - st.session_state.last_phrase_time > 1.0:  # 1 second delay
            phrase = gesture_to_phrase[gesture_text]
            if st.session_state.current_sentence:
                st.session_state.current_sentence += " " + phrase
            else:
                st.session_state.current_sentence += phrase
            st.session_state.last_phrase_time = current_time
            return f"Added: {phrase}"

    # Process space gesture
    elif gesture_text == "␣ SPACE":
        if current_time - st.session_state.last_phrase_time > 1.0:
            if st.session_state.current_sentence and not st.session_state.current_sentence.endswith(" "):
                st.session_state.current_sentence += " "
                st.session_state.last_phrase_time = current_time
                return "Space added"

    # Process delete gesture
    elif gesture_text == "⌫ DELETE":
        if current_time - st.session_state.last_phrase_time > 1.0:
            if st.session_state.current_sentence:
                if st.session_state.current_sentence.endswith(" "):
                    st.session_state.current_sentence = st.session_state.current_sentence.rstrip()
                else:
                    words = st.session_state.current_sentence.split()
                    if words:
                        st.session_state.current_sentence = " ".join(words[:-1])
                st.session_state.last_phrase_time = current_time
                return "Last phrase deleted"

    # Process enter/confirm gesture
    elif gesture_text == "↵ ENTER":
        if current_time - st.session_state.last_phrase_time > 1.5:
            if st.session_state.current_sentence:
                completed_sentence = st.session_state.current_sentence + "."
                st.session_state.sentence_history.append(completed_sentence)
                st.session_state.current_sentence = ""
                st.session_state.last_phrase_time = current_time
                return f"Sentence completed: {completed_sentence}"

    return None

# --- 7. Helper Functions (Data Collection) ---

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

# --- 8. Sidebar UI ---

current_samples = load_current_samples()
total_samples_collected = sum(current_samples.values())

with col_sidebar:
    # --- Theme Toggle ---
    if st.button("Toggle Theme"):
        st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
        apply_theme_css()

    # --- About Signo Expander ---
    with st.expander("About Signo Sentence Builder", expanded=False):
        st.markdown("""
        **Sentence Building Gestures:**
        - **␣ SPACE**: Flat hand with thumb out (add space)
        - **⌫ DELETE**: Thumb-index pinch (delete last phrase)
        - **↵ ENTER**: Thumbs up (finish sentence with period)
        - **Phrases/Letters**: Recognized signs
        
        **Usage:** Use gestures to build sentences with phrases and letters!
        """)
    st.markdown("---")

    # --- Sentence Builder Section ---
    st.subheader("📝 Sentence Builder")

    # Current sentence display
    st.text_area("Current Sentence",
                value=st.session_state.current_sentence or "Start building...",
                height=100,
                key="current_sentence_display")

    # Sentence history
    if st.session_state.sentence_history:
        st.write("**Recent Sentences:**")
        for i, sentence in enumerate(reversed(st.session_state.sentence_history[-5:])):
            st.code(f"{len(st.session_state.sentence_history)-i}. {sentence}")

    # Control buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("␣ Space", use_container_width=True):
            if st.session_state.current_sentence and not st.session_state.current_sentence.endswith(" "):
                st.session_state.current_sentence += " "
    with col2:
        if st.button("⌫ Delete", use_container_width=True):
            if st.session_state.current_sentence:
                if st.session_state.current_sentence.endswith(" "):
                    st.session_state.current_sentence = st.session_state.current_sentence.rstrip()
                else:
                    words = st.session_state.current_sentence.split()
                    if words:
                        st.session_state.current_sentence = " ".join(words[:-1])
    with col3:
        if st.button("↵ Finish Sentence", use_container_width=True):
            if st.session_state.current_sentence:
                st.session_state.sentence_history.append(st.session_state.current_sentence + ".")
                st.session_state.current_sentence = ""
    
    st.markdown("---")
    
    # --- Samples Metric ---
    st.metric(label="Total Data Samples Collected", value=f"{total_samples_collected:,}")
    st.markdown("---")

    # --- Mode Selection ---
    st.header("Application Controls")
    
    app_mode = st.radio(
        "Select Operation Mode:",
        ("Sentence Builder Mode (Live Demo)", "Data Collection Mode (ML Prep)"),
        index=0
    )

    st.markdown("---")

    if app_mode == "Sentence Builder Mode (Live Demo)":
        st.subheader("Sentence Builder Mode")
        st.info("""
        **Gesture Guide:**
        - ✋␣ **Space**: Flat hand, thumb out (add space)
        - 👌⌫ **Delete**: Thumb-index pinch (delete last phrase)
        - 👍↵ **Enter**: Thumbs up (finish sentence)
        - Use letters and phrases to build sentences!
        """)
        
    else:  # Data Collection Mode
        st.subheader("Data Collection Setup")
        selected_letter = st.selectbox("Select KSL Sign to Record:", KSL_LETTERS)
        num_samples_to_record = st.slider(
            f"Target Samples for '{selected_letter}':",
            min_value=50, max_value=500, value=200, step=10
        )
        
        if 'recording' not in st.session_state:
            st.session_state.recording = False
        
        button_label = "Stop Recording" if st.session_state.recording else "Start Recording"
        if st.button(button_label, use_container_width=True):
            st.session_state.recording = not st.session_state.recording

# --- 9. Main Video Feed ---

with col_main:
    st.subheader("Real-Time KSL Sentence Building")
    
    run = st.checkbox("Start Webcam Feed", value=False)
    FRAME_WINDOW = st.image([])
    
    # Feedback container for word building actions
    FEEDBACK_CONTAINER = st.empty()

# --- 10. Main Webcam Loop ---

cap = None
prev_time = 0

if run:
    with st.spinner("Starting webcam..."):
        cap = cv2.VideoCapture(0)
        
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
        
        apply_theme_css()
        
        # Clear old feedback messages
        if st.session_state.feedback_display_time > 0 and current_time > st.session_state.feedback_display_time:
            FEEDBACK_CONTAINER.empty()
            st.session_state.feedback_display_time = 0

        # Process frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)
        
        frame.flags.writeable = False
        result = hands.process(frame)
        frame.flags.writeable = True

        gesture_text = "No hand detected"
        word_feedback = None
        
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(240, 240, 240), thickness=2, circle_radius=2), 
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                )
                
                landmarks_list = hand_landmarks.landmark

                if app_mode == "Sentence Builder Mode (Live Demo)":
                    gesture_text = classify_sign_geometric(landmarks_list)

                    # Handle sentence building
                    word_feedback = handle_sentence_building(gesture_text, current_time)

                    # Show sentence building feedback
                    if word_feedback:
                        FEEDBACK_CONTAINER.success(word_feedback, icon="✅")
                        st.session_state.feedback_display_time = current_time + 2.0
                        
                elif app_mode == "Data Collection Mode (ML Prep)" and st.session_state.recording:
                    features = extract_landmarks(result.multi_hand_landmarks)
                    save_data(selected_letter, features)
                    gesture_text = f"Recording '{selected_letter}'"

        # Determine border color
        if app_mode == "Sentence Builder Mode (Live Demo)":
            border_bgr = (255, 100, 0)  # Blue for Sentence Builder
        elif app_mode == "Data Collection Mode (ML Prep)" and st.session_state.recording:
            border_bgr = (0, 0, 255)    # Red for Recording
        else:
            border_bgr = (0, 255, 0)    # Green for Ready

        # Draw dynamic border
        border_thickness = 15
        H, W, _ = frame.shape
        cv2.rectangle(frame, (0, 0), (W, H), border_bgr, border_thickness)

        # Draw status overlays
        # FPS counter (green)
        cv2.putText(frame, f'FPS: {int(fps)}', (25, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Gesture hints (when in sentence builder mode)
        hint_y = 90
        cv2.putText(frame, "👍 Enter", (25, hint_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, "👊 Delete", (25, hint_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, "✋ Space", (25, hint_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

        # Current sentence display on video
        sentence_display = f"Sentence: {st.session_state.current_sentence}" if st.session_state.current_sentence else "Start building sentences..."
        sentence_text_size = cv2.getTextSize(sentence_display, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        sentence_x = 25
        sentence_y = H - 80
        cv2.rectangle(frame, (sentence_x-10, sentence_y-sentence_text_size[1]-10),
                     (sentence_x+sentence_text_size[0]+10, sentence_y+10), (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, sentence_display, (sentence_x, sentence_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Gesture text overlay
        text_size = cv2.getTextSize(gesture_text, cv2.FONT_HERSHEY_DUPLEX, 1.2, 2)[0]
        text_x = (W - text_size[0]) // 2
        text_y = H - 30
        cv2.rectangle(frame, (text_x-10, text_y-text_size[1]-10), 
                     (text_x+text_size[0]+10, text_y+10), (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, gesture_text, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

        FRAME_WINDOW.image(frame, channels="RGB")
        
        if not run:
            break
        
        time.sleep(0.01)

# Cleanup
if cap and cap.isOpened():
    cap.release()
    hands.close()

if not run:
    FRAME_WINDOW.empty()
    with col_main:
        st.subheader("Real-Time KSL Sentence Building")
        st.info("Click 'Start Webcam Feed' to begin building sentences with gestures!")
