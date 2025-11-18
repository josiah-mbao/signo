# 🤖 Signo: Kenya Sign Language (KSL) Sentence & Phrase Builder

<img width="1440" height="900" alt="signo sc1" src="https://github.com/user-attachments/assets/cc5f78fc-c426-42c0-95a1-2e0a662a3396" />


**Signo** is a real-time, interactive hand gesture recognition application designed for Kenya Sign Language (KSL). Built with **Streamlit**, **OpenCV**, and **MediaPipe**, it uses geometric classification to recognize individual letters and common phrases, enabling users to build sentences directly from webcam gestures.



---

## ✨ Features

* **Sentence Building:** Combine recognized letters and phrases to build complete sentences using intuitive gestures.
* **Phrase Recognition:** Recognizes common full words/phrases alongside individual KSL letters.
* **Real-Time Recognition:** Processes live video from your webcam to detect and classify hand gestures instantly.
* **Robust Classification:** Employs geometric checks on finger joint positions for reliable gesture identification.
* **Interactive UI:** Sidebar displays current sentence, history, and manual controls; supports light/dark theme toggle.
* **Data Collection Mode:** Collect hand landmark data for ML model training.
* **Visual Feedback:** Overlays MediaPipe landmarks and gesture text on live video feed.

---

## 🖐️ Supported Gestures

The application recognizes KSL letters and phrases:

**Letters:**
- ✊ S, ✋ B, 🅰️ A, 🤟 L, ✌️ V, 🤙 Y, 🖕 I

**Phrases:**
- HELLO (Flat hand with thumb left), THANK YOU (Fist with thumb right extended)

**Building Gestures:**
- ␣ SPACE: Flat hand with thumb right (add space)
- ⌫ DELETE: Thumb-index pinch (delete last phrase)
- ↵ ENTER: Thumbs up (finish sentence with period)

---

## ⚙️ Installation and Setup

### Prerequisites

You need **Python 3.7+** installed on your system.

### Steps

1.  **Clone the Repository (if applicable):**
    ```bash
    git clone <your-repository-url>
    cd signo-gesture-recognition
    ```

2.  **Install Dependencies:**
    The application relies on `streamlit`, `opencv-python`, and `mediapipe`.

    ```bash
    pip install streamlit opencv-python mediapipe numpy
    ```

3.  **Run the Application:**
    Execute the main script using Streamlit.

    ```bash
    streamlit run final.py
    ```

    The application will launch in your default web browser (usually at `http://localhost:8501`).

---

## 💻 Technology Stack

* **Core Language:** Python
* **Web Framework:** [Streamlit](https://streamlit.io/)
* **Hand Tracking:** [Google MediaPipe Hands](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)
* **Video Processing:** [OpenCV (`cv2`)](https://opencv.org/)
* **Numerical Operations:** [NumPy](https://numpy.org/)

---

## 🛠️ Classification Details (For Developers)

The core logic resides in the `is_finger_open` and `classify_gesture` functions.

1.  **Finger Open Check (`is_finger_open`):**
    A finger is considered **open** if its **tip's** $y$-coordinate is *smaller* than its corresponding **PIP** (Proximal Interphalangeal) joint's $y$-coordinate.
    * *Rationale:* In image coordinates, the $y$-axis increases downwards. When a finger is extended vertically, the tip is positioned "higher" (smaller $y$ value) than the joint closer to the palm. This makes the detection robust regardless of hand size or distance from the camera.

2.  **Gesture Mapping (`classify_gesture`):**
    This function checks the combined state (open/closed) of the four non-thumb fingers (Index, Middle, Ring, Pinky) to match them against the predefined gesture patterns. A special case is implemented for **👍 Thumbs Up** which requires both a specific finger state *and* the thumb tip to be positioned above the wrist.

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
