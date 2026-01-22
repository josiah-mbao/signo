# 🤖 Signo: Interactive Gesture-Based Sentence Builder

<img width="1440" height="900" alt="signo sc1" src="https://github.com/user-attachments/assets/cc5f78fc-c426-42c0-95a1-2e0a662a3396" />


**Signo** is an interactive gesture recognition application demonstrating sentence building through hand gestures. Built with **Streamlit** and **OpenCV**, it uses geometric pattern recognition to identify hand shapes mapped to common words and phrases, allowing users to construct sentences through intuitive gesture interactions.

**Note:** This application uses simplified geometric recognition for educational demonstration purposes. It does not perform authentic ASL (American Sign Language) recognition or real-time hand tracking.

## 🎯 Try It Now (2-Minute Demo)

**Want to see ASL gesture recognition in action?**

```bash
git clone https://github.com/josiah-mbao/signo.git
cd signo
./run_app.sh
```

Then visit: **http://localhost:8503**

**What you'll see:**
- 🎥 **Real-time webcam gesture recognition**
- ✋ **Build sentences with hand gestures** (SPACE, DELETE, ENTER)
- 📝 **Recognize letters A, B, I, L, S, V, Y and phrases** (HELLO, THANK YOU, GOOD DAY)
- 🎨 **Interactive UI with theme toggle**

**🚀 Live Demo:** [Signo on Streamlit Cloud](https://signo.streamlit.app) *(Interface preview - webcam requires local deployment)*

## 🆕 Recent Updates

**Version 2.1 - Gesture Recognition Focus**
- ✅ **Geometric Pattern Recognition:** Simplified, reliable gesture detection without external dependencies
- ✅ **Demo Gesture Cycling:** Time-based gesture simulation for educational demonstration
- ✅ **Enhanced Webcam Support:** Improved camera detection and macOS compatibility
- ✅ **Accurate Documentation:** README updated to reflect actual capabilities and limitations
- ✅ **Deployment Ready:** Streamlit Cloud compatible with automatic environment detection

---

## ✨ Features

* **Gesture-Based Sentence Building:** Construct complete sentences using intuitive hand gestures and geometric pattern recognition.
* **Demo Gesture Recognition:** Recognizes simplified hand shapes mapped to common words and phrases for educational demonstration.
* **Interactive Sentence Construction:** Build sentences through gesture patterns that cycle through different words and control commands.
* **Real-Time Visual Feedback:** Live webcam feed with gesture recognition overlays and sentence building progress.
* **Manual Control Options:** Sidebar buttons provide alternative input methods alongside gesture recognition.
* **Theme Customization:** Light/dark mode toggle for comfortable viewing in different environments.
* **Educational Interface:** Designed for HCI education, demonstrating gesture-based interaction concepts.
* **Responsive Design:** Clean, modern UI that works across different screen sizes and devices.

---



## 🖐️ Assigned Gesture Mappings

The application recognizes hand shapes assigned to common words and phrases for interactive sentence building (these are **not authentic ASL signs** but demo gestures for HCI education):

**Word Shapes:**
- ✊ "I" (Fist), ✋ "YOU" (Flat hand), 🅰️ "AM" (A shape), 🤟 "GOOD" (L shape)
- ✌️ "WE" (V shape), 🤙 "SEE" (Y shape), 🖕 "ME" (I shape)

**Phrase Shapes:**
- HELLO (Flat hand with thumb left), THANK YOU (Fist with thumb right extended)
- GOOD DAY (V shape with thumb added)

**Control Gestures:**
- ✋ SPACE: Flat hand with thumb right (add space between words)
- 👊 DELETE: Thumb-index pinch (remove last word/phrase)
- 👍 ENTER: Thumbs up (complete sentence with period)

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
    The application relies on `streamlit` and `opencv-python` for core functionality.

    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application:**
    Execute the main script using Streamlit.

    ```bash
    streamlit run final.py
    ```

    The application will launch in your default web browser (usually at `http://localhost:8501`).

### 🚀 Quick Start (Single Command)

For the easiest setup, use the automated script that handles everything:

```bash
# Make sure you're in the project directory
cd signo

# Run the app locally with one command
./run_app.sh
```

This script will:
- ✅ Install Python dependencies automatically
- ✅ Start the Streamlit application locally
- ✅ Display the local access URL (http://localhost:8503)

**Perfect for local development and testing!** Just run `./run_app.sh` and access at `http://localhost:8503`.

For sharing with others over the internet, you'll need to use ngrok or similar tunneling service separately:
```bash
# After starting the app with ./run_app.sh
ngrok http 8503  # This will give you a public URL
```

---

## 🚀 Deployment

### Streamlit Cloud Deployment

1. **Fork this repository** to your GitHub account

2. **Connect to Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select this repository
   - Set main file path to `final.py`
   - Click Deploy

3. **Access your deployed app** via the provided URL

**Note:** Webcam functionality is not available in Streamlit Cloud due to browser security restrictions. Users can see the interface and interact with the sentence builder controls.

### Local Webcam Demo

For full functionality including webcam gesture recognition:

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally with webcam access
streamlit run final.py
```

---

## 💻 Technology Stack

* **Core Language:** Python 3.11+
* **Web Framework:** [Streamlit](https://streamlit.io/) with custom theming and responsive UI
* **Gesture Recognition:** Geometric pattern matching using hand shape analysis
* **Video Processing:** [OpenCV](https://opencv.org/) for real-time video capture and processing
* **Numerical Operations:** [NumPy](https://numpy.org/) for efficient array operations and geometric calculations
* **Demo Cycling:** Time-based gesture simulation for educational demonstration
* **UI Components:** Custom Streamlit interface with real-time feedback and controls

---

## 🛠️ Classification Details (For Developers)

The core logic resides in the `is_finger_open` and `classify_gesture` functions.

1.  **Finger Open Check (`is_finger_open`):**
    A finger is considered **open** if its **tip's** $y$-coordinate is *smaller* than its corresponding **PIP** (Proximal Interphalangeal) joint's $y$-coordinate.
    * *Rationale:* In image coordinates, the $y$-axis increases downwards. When a finger is extended vertically, the tip is positioned "higher" (smaller $y$ value) than the joint closer to the palm. This makes the detection robust regardless of hand size or distance from the camera.

2.  **Gesture Mapping (`classify_gesture`):**
    This function checks the combined state (open/closed) of the four non-thumb fingers (Index, Middle, Ring, Pinky) to match them against the predefined gesture patterns. A special case is implemented for **👍 Thumbs Up** which requires both a specific finger state *and* the thumb tip to be positioned above the wrist.

3.  **Model Training & Evaluation:**
    The `asl_trainer.py` module provides comprehensive model training with:
    - Cross-validation for robust performance estimation
    - Confusion matrix analysis for error patterns
    - Per-class accuracy breakdown
    - Automatic metrics export for UI display

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
