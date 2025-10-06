import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

# --- 1. Streamlit Configuration and Layout ---
st.set_page_config(page_title="Signo - Hand Gesture Demo", page_icon="✋", layout="wide")

st.title("🤖 Signo: Robust Hand Gesture Recognition Demo")
st.write("Show your hand to the webcam and see the recognized gesture in real-time!")

# Use columns for a centered layout
col_main, col_sidebar = st.columns([4, 1.5]) 

with col_sidebar:
    st.markdown("---")
    st.header("Supported Gestures")
    st.markdown("""
        - **✋ Open Palm**: All fingers extended.
        - **✊ Fist**: All fingers curled.
        - **👍 Thumbs Up**: Thumb extended, other fingers curled.
        - **✌️ Peace Sign**: Index and Middle fingers extended.
        - **🤘 Rock On / ILY**: Index and Pinky fingers extended.
        - **🤷 Unknown**: Anything else.
    """)
    st.markdown("---")
    st.info("The classification uses relative $y$-coordinate comparisons of finger tips and joints for better robustness.")

# Placeholder for the central content (video and status)
with col_main:
    # Use nested columns to center the video within the main column
    _, col_video, _ = st.columns([1, 6, 1])
    
    with col_video:
        run = st.checkbox("Start Webcam", value=False)
        FRAME_WINDOW = st.image([])
        # Dedicated placeholder for the gesture status text
        gesture_status = st.empty()

# --- 2. Mediapipe Setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.7
)

# --- 3. Robust Classification Logic ---

def is_finger_open(landmarks, tip_id, pip_id, mcp_id=None):
    """
    Checks if a finger is open by comparing the y-coordinate of the tip 
    to the y-coordinate of the PIP joint. Y increases downwards in image coordinates.
    Tip must be above PIP (smaller Y value) to be considered open.
    """
    tip_y = landmarks[tip_id].y
    pip_y = landmarks[pip_id].y
    
    # Threshold check: tip is significantly above the PIP joint
    return tip_y < pip_y

def classify_gesture(landmarks):
    """
    Classifies a gesture based on the relative position of finger landmarks.
    Landmark Indices: [0:Wrist, 4:Thumb Tip, 8:Index Tip, 12:Middle Tip, 16:Ring Tip, 20:Pinky Tip]
    PIP Joints: [3:Thumb IP, 6:Index PIP, 10:Middle PIP, 14:Ring PIP, 18:Pinky PIP]
    """
    # Key landmark indices (Tip, PIP)
    # Note: Using IP for thumb (3) and PIP for others.
    FINGERS = {
        'thumb': (4, 3), 
        'index': (8, 6), 
        'middle': (12, 10), 
        'ring': (16, 14), 
        'pinky': (20, 18)
    }
    
    fingers_open = {name: is_finger_open(landmarks, tip, pip) 
                    for name, (tip, pip) in FINGERS.items()}
    
    # ✊ Fist: All non-thumb fingers closed (and thumb tucked/closed)
    if (not fingers_open['index'] and not fingers_open['middle'] and 
        not fingers_open['ring'] and not fingers_open['pinky']):
        return "✊ Fist"

    # ✋ Open Palm: All non-thumb fingers open
    if (fingers_open['index'] and fingers_open['middle'] and 
        fingers_open['ring'] and fingers_open['pinky']):
        return "✋ Open Palm"
        
    # 👍 Thumbs Up: Thumb open, others closed (using a simple X-check for thumb position)
    if fingers_open['thumb'] and (not fingers_open['index'] and not fingers_open['middle']):
        # Ensure the thumb tip is relatively high (y-coord < wrist y) for thumbs up
        if landmarks[4].y < landmarks[0].y:
            return "👍 Thumbs Up"

    # ✌️ Peace Sign: Index and Middle open, others closed
    if (fingers_open['index'] and fingers_open['middle'] and 
        not fingers_open['ring'] and not fingers_open['pinky']):
        return "✌️ Peace Sign"

    # 🤘 Rock On / ILY: Index and Pinky open, Middle/Ring closed
    if (fingers_open['index'] and fingers_open['pinky'] and 
        not fingers_open['middle'] and not fingers_open['ring']):
        return "🤘 Rock On / ILY"

    return "🤷 Unknown"

# --- 4. Main Webcam Loop ---

cap = None

if run:
    with st.spinner("Starting webcam..."):
        # Initialize video capture inside the 'run' block
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        
        if not cap.isOpened():
            st.error("Failed to access webcam. Check permissions and try restarting.")
            run = False
            
    while run and cap.isOpened():
        # Re-check the state of the checkbox inside the loop
        # (This relies on Streamlit's rerunning behavior to break the loop cleanly)
        if not st.session_state.get('Start Webcam', True): 
             break
             
        ret, frame = cap.read()
        if not ret:
            gesture_status.error("Lost webcam feed.")
            break

        # Convert the frame to RGB for MediaPipe processing
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Flip the frame horizontally for a natural 'mirror' view
        frame = cv2.flip(frame, 1) 
        
        # Set frame as not writeable to improve performance
        frame.flags.writeable = False
        result = hands.process(frame)
        frame.flags.writeable = True

        gesture = "No hand detected"
        
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw landmarks on the frame
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(240, 240, 240), thickness=2, circle_radius=2), # white
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2) # green
                )
                
                # Classify the gesture
                gesture = classify_gesture(hand_landmarks.landmark)

            gesture_status.success(f"## Recognized: **{gesture}**")
        else:
            gesture_status.info("## Recognized: *No hand detected*")

        # Display the processed frame
        FRAME_WINDOW.image(frame, channels="RGB")

# --- 5. Cleanup ---
if cap and cap.isOpened():
    cap.release()
    hands.close()
    gesture_status.warning("Webcam stopped.")
    
if not run:
    FRAME_WINDOW.empty()
    gesture_status.empty()
    gesture_status.info("Click 'Start Webcam' to begin.")

