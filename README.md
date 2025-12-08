# 🤖 Signo: Interactive Gesture-Based Sentence Builder

<img width="1440" height="900" alt="signo sc1" src="https://github.com/user-attachments/assets/cc5f78fc-c426-42c0-95a1-2e0a662a3396" />


**Signo** is a real-time, interactive hand gesture recognition application demonstrating sentence building through gesture input. Built with **Streamlit**, **OpenCV**, and **MediaPipe**, it uses geometric classification to recognize assigned hand shapes mapped to common words and phrases, enabling users to build sentences directly from webcam gestures.

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
- 📝 **Recognize letters A-Z and phrases** (HELLO, THANK YOU, etc.)
- 🎨 **Interactive UI with theme toggle**

**🚀 Live Demo:** [Signo on Streamlit Cloud](https://signo.streamlit.app) *(Interface preview - webcam requires local deployment)*

## 🆕 Recent Updates

**Version 2.0 - Enterprise Metrics & Deployment**
- ✅ **Advanced Model Evaluation:** Cross-validation, confusion matrices, per-class accuracy analysis
- ✅ **Real-time Metrics Dashboard:** Interactive visualizations integrated into Streamlit UI
- ✅ **Multi-Model Comparison:** SVM, Random Forest, and Logistic Regression support
- ✅ **Deployment Ready:** Streamlit Cloud compatible with automatic environment detection
- ✅ **Enhanced Documentation:** Comprehensive README with deployment guides

---

## ✨ Features

* **Sentence Building:** Combine recognized letters and phrases to build complete sentences using intuitive gestures.
* **Phrase Recognition:** Recognizes common full words/phrases alongside individual KSL letters.
* **Real-Time Recognition:** Processes live video from your webcam to detect and classify hand gestures instantly.
* **Robust Classification:** Employs geometric checks on finger joint positions for reliable gesture identification.
* **Interactive UI:** Sidebar displays current sentence, history, and manual controls; supports light/dark theme toggle.
* **Data Collection Mode:** Collect hand landmark data for ML model training with real-time feedback.
* **Visual Feedback:** Overlays MediaPipe landmarks, gesture text, and control hints (👍👊✋) on live video feed.
* **Demo-Ready UI:** Polished interface with webcam overlay hints for clean presentations.
* **Enterprise Metrics:** Advanced model evaluation with confusion matrices, cross-validation, and per-class accuracy analysis.
* **Deployment Ready:** Streamlit Cloud compatible with automatic environment detection and fallback interfaces.
* **Multi-Model Support:** Compare SVM, Random Forest, and Logistic Regression performance.
* **Real-Time Analytics:** Live metrics dashboard with interactive visualizations and performance tracking.

---

## 📊 Advanced Model Performance Metrics

The application now includes enterprise-grade evaluation metrics for trained ASL classifiers:

### Core Metrics Dashboard
- **Test Accuracy:** Overall model performance on held-out test data
- **Cross-Validation Scores:** 5-fold stratified CV with mean and standard deviation
- **Per-Class Accuracy:** Individual accuracy breakdown for all 26 ASL letters (A-Z)

### Advanced Analytics
- **Confusion Matrix:** Visual heatmap showing prediction patterns and error distributions
- **Classification Report:** Detailed precision, recall, F1-scores, and support metrics
- **Model Robustness:** Cross-validation confidence intervals for reliable performance estimation

### Real-time Metrics Display
- Interactive bar charts showing per-class accuracy distributions
- Confusion matrix visualization with error pattern analysis
- Live metrics updates integrated into the Streamlit interface

---

## 🖐️ Assigned Gesture Mappings

The application recognizes hand shapes assigned to common words and phrases for interactive sentence building (these are **not authentic KSL signs** but demo gestures for HCI education):

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
    The application relies on `streamlit`, `opencv-python`, and `mediapipe`.

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

* **Core Language:** Python 3.7+
* **Web Framework:** [Streamlit](https://streamlit.io/) with custom theming and responsive UI
* **Hand Tracking:** [Google MediaPipe Hands](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)
* **Video Processing:** [OpenCV](https://opencv.org/) for real-time video capture and processing
* **Numerical Operations:** [NumPy](https://numpy.org/) for efficient array operations
* **Machine Learning:** [Scikit-learn](https://scikit-learn.org/) with SVM, Random Forest, and Logistic Regression
* **Model Evaluation:** Advanced metrics including cross-validation, confusion matrices, and per-class analysis
* **Data Visualization:** [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/) for metrics visualization
* **Data Processing:** [Pandas](https://pandas.pydata.org/) for dataset management and analysis

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
