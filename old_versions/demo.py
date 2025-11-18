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

if not os.path.exists(DATA_PATH):
    header = ['label'] + [f'landmark_{i}_{coord}' for i in range(21) for coord in ['x', 'y', 'z']]
    with open(DATA_PATH, mode='w', newline='') as f:
        csv.writer(f).writerow(header)

# --- 2. Streamlit Configuration ---
st.set_page_config(page_title="Signo - KSL Data Collection & Demo", page_icon="✋", layout="wide")
st.title("🇰🇪 Signo: Kenya Sign Language (KSL) Gesture Translator Demo")

col_main, col_sidebar = st.columns([4, 1.5])

# --- 3. Theme Toggle ---
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

def apply_theme_css():
    """Inject CSS for dynamic light/dark mode."""
    if st.session_state.theme == 'dark':
        css = """
        <style>
            [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
                background-color: #0E1117 !important;
                color: #FFFFFF !important;
            }
            h1, h2, h3, h4, h5, h6, label, span, p, div[data-testid="stText"], div[data-testid="stMarkdown"] {
                color: #FFFFFF !important;
            }
            .stMetric {
                color: #FFFFFF !important;
            }
        </style>
        """
    else:
        css = """
        <style>
            [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
                background-color: #F7F9FF !important;
                color: #262730 !important;
            }
            h1, h2, h3, h4, h5, h6, label, span, p, div[data-testid="stText"], div[data-testid="stMarkdown"] {
                color: #262730 !important;
            }
            .stMetric {
                color: #262730 !important;
            }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)

with col_sidebar:
    st.radio(
        "Select Theme:",
        ["light", "dark"],
        index=0 if st.session_state.theme == 'light' else 1,
        key="theme_toggle",
        on_change=lambda: [st.session_state.update({'theme': st.session_state.theme_toggle}), apply_theme_css()]
    )

apply_theme_css()

# --- 4. Mediapipe Setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.6
)

# --- 5. Geometric Classification ---
def is_finger_open(landmarks, tip_id, pip_id):
    return landmarks[tip_id].y < landmarks[pip_id].y

def classify_sign_geometric(landmarks):
    FINGERS = {
        'thumb': (4, 3), 'index': (8, 6), 'middle': (12, 10), 
        'ring': (16, 14), 'pinky': (20, 18)
    }
    fingers_open = {name: is_finger_open(landmarks, tip, pip) for name, (tip, pip) in FINGERS.items()}
    thumb_tip_x = landmarks[4].x
    index_mcp_x = landmarks[5].x

    if all(not fingers_open[f] for f in ['index','middle','ring','pinky']):
        if landmarks[4].y > landmarks[3].y:
            return "✊ FIST / KSL 'S'"
    if all(fingers_open[f] for f in ['index','middle','ring','pinky']):
        if not fingers_open['thumb'] and landmarks[4].x > landmarks[5].x:
            return "✋ OPEN PALM / KSL 'B'"
    if all(not fingers_open[f] for f in ['index','middle','ring','pinky']):
        if fingers_open['thumb'] and thumb_tip_x < index_mcp_x:
            return "🅰️ KSL 'A'"
    if fingers_open['index'] and fingers_open['thumb'] and not any([fingers_open['middle'],fingers_open['ring'],fingers_open['pinky']]):
        return "🤟 KSL 'L'"
    if fingers_open['index'] and fingers_open['middle'] and not fingers_open['ring'] and not fingers_open['pinky']:
        return "✌️ KSL 'V'"
    if fingers_open['thumb'] and fingers_open['pinky'] and not any([fingers_open['index'], fingers_open['middle'], fingers_open['ring']]):
        return "🤙 KSL 'Y'"
    if fingers_open['pinky'] and not any([fingers_open['thumb'], fingers_open['index'], fingers_open['middle'], fingers_open['ring']]):
        return "🖕 KSL 'I'"
    return "🤷 Unknown"

# --- 6. Helper Functions ---
@st.cache_data
def load_current_samples():
    if not os.path.exists(DATA_PATH):
        return {}
    try:
        data = np.loadtxt(DATA_PATH, delimiter=',', skiprows=1, dtype=str)
        if data.ndim == 1 and data.size > 0:
            data = data.reshape(1,-1)
        if data.size == 0:
            return {}
        labels = data[:,0]
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique, counts))
    except Exception:
        return {}

def extract_landmarks(hand_landmarks):
    if not hand_landmarks:
        return None
    data = []
    for lm in hand_landmarks[0].landmark:
        data.extend([lm.x, lm.y, lm.z])
    return data

def save_data(label, features):
    if features is None:
        return 0
    row = [label] + features
    with open(DATA_PATH, 'a', newline='') as f:
        csv.writer(f).writerow(row)
    return 1

# --- 7. Sidebar UI ---
current_samples = load_current_samples()
total_samples_collected = sum(current_samples.values())

with col_sidebar:
    st.metric("Total Data Samples Collected", f"{total_samples_collected:,}")

    app_mode = st.radio(
        "Select Operation Mode:",
        ("Quick Demo Mode (Geometric Rules)", "Data Collection Mode (ML Prep)"),
        index=0
    )

    if app_mode == "Data Collection Mode (ML Prep)":
        selected_letter = st.selectbox("Select KSL Sign to Record:", KSL_LETTERS)
        num_samples_to_record = st.slider(
            f"Target Samples for '{selected_letter}'",
            min_value=50, max_value=500, value=200, step=10
        )

        if 'recording' not in st.session_state:
            st.session_state.recording = False

        button_label = "Stop Recording" if st.session_state.recording else "Start Recording"
        if st.button(button_label):
            st.session_state.recording = not st.session_state.recording
            st.experimental_rerun()

# --- 8. Main Video Feed ---
with col_main:
    st.subheader("Real-Time KSL Gesture Recognition")
    run = st.checkbox("Start Webcam Feed", value=False)
    FRAME_WINDOW = st.image([])

cap = None
prev_time = 0

if run:
    cap = cv2.VideoCapture(0, cv2.CAP_ANY)
    while run and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Lost webcam feed.")
            break

        current_time = time.time()
        fps = 1 / (current_time - prev_time) if current_time != prev_time else 0
        prev_time = current_time

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)
        frame.flags.writeable = False
        result = hands.process(frame)
        frame.flags.writeable = True

        gesture_text = "No hand detected"
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks_list = hand_landmarks.landmark
                if app_mode == "Quick Demo Mode (Geometric Rules)":
                    gesture_text = classify_sign_geometric(landmarks_list)
                elif app_mode == "Data Collection Mode (ML Prep)" and st.session_state.recording:
                    features = extract_landmarks(result.multi_hand_landmarks)
                    save_data(selected_letter, features)
                    gesture_text = f"Recording '{selected_letter}'"

        # Border
        border_color = (255, 100, 0) if app_mode.startswith("Quick") else (0, 0, 255) if st.session_state.get('recording', False) else (0, 255, 0)
        cv2.rectangle(frame, (0,0), (frame.shape[1], frame.shape[0]), border_color, 15)

        # FPS counter (green)
        cv2.putText(frame, f'FPS: {int(fps)}', (25, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

        # Gesture overlay
        text_size = cv2.getTextSize(gesture_text, cv2.FONT_HERSHEY_DUPLEX, 1.5, 3)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = frame.shape[0] - 45
        cv2.rectangle(frame, (text_x-10, text_y-text_size[1]-10), (text_x+text_size[0]+10, text_y+10), (0,0,0), cv2.FILLED)
        cv2.putText(frame, gesture_text, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255,255,255), 3, cv2.LINE_AA)

        FRAME_WINDOW.image(frame, channels="RGB")
        time.sleep(0.01)

# Cleanup
if cap and cap.isOpened():
    cap.release()
    hands.close()
