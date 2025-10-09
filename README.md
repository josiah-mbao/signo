# 🤖 Signo: Robust Hand Gesture Recognition Demo

**Signo** is a real-time, interactive hand gesture recognition application built with **Streamlit**, **OpenCV**, and **MediaPipe**. It leverages a robust, custom-defined classification logic based on finger joint and tip positions to reliably identify common hand signs directly from a webcam feed.



---

## ✨ Features

* **Real-Time Recognition:** Processes live video from your webcam to detect and classify hand gestures instantly.
* **Robust Classification:** Employs a relative $y$-coordinate comparison of finger landmarks (tips vs. PIP/IP joints) for gesture identification, making the system less sensitive to slight variations in hand angle and distance.
* **Clear Visual Feedback:** Overlays MediaPipe's landmark connections onto the video feed, providing clear visualization of the hand tracking.
* **Streamlit Interface:** Provides a simple, modern, and cross-platform web interface with an easy-to-use **Start/Stop Webcam** control.

---

## 🖐️ Supported Gestures

The application is programmed to accurately distinguish between the following hand signs:

| Gesture | Description | Classification Logic |
| :---: | :--- | :--- |
| **✋ Open Palm** | All fingers extended (open). | All non-thumb fingers are open. |
| **✊ Fist** | All fingers curled (closed). | All non-thumb fingers are closed. |
| **👍 Thumbs Up** | Thumb extended, others curled. | Thumb is open, Index/Middle are closed, and the thumb tip is high relative to the wrist. |
| **✌️ Peace Sign** | Index and Middle fingers extended. | Index and Middle fingers are open, Ring and Pinky are closed. |
| **🤘 Rock On / ILY** | Index and Pinky fingers extended. | Index and Pinky fingers are open, Middle and Ring are closed. |
| **🤷 Unknown** | Any other configuration. | Default return for unclassified states. |

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
    Execute the Python script using Streamlit.

    ```bash
    streamlit run app.py # Replace 'app.py' with your script's filename
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
