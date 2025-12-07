#!/usr/bin/env python3
"""
Test script to check if all required imports work correctly
"""

def test_imports():
    print("Testing imports...")

    try:
        import streamlit as st
        print(f"✅ Streamlit {st.__version__}")
    except ImportError as e:
        print(f"❌ Streamlit: {e}")

    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy: {e}")

    try:
        import pandas as pd
        print(f"✅ Pandas {pd.__version__}")
    except ImportError as e:
        print(f"❌ Pandas: {e}")

    try:
        import sklearn
        print(f"✅ Scikit-learn {sklearn.__version__}")
    except ImportError as e:
        print(f"❌ Scikit-learn: {e}")

    try:
        import matplotlib.pyplot as plt
        print(f"✅ Matplotlib {plt.matplotlib.__version__}")
    except ImportError as e:
        print(f"❌ Matplotlib: {e}")

    try:
        import seaborn as sns
        print(f"✅ Seaborn {sns.__version__}")
    except ImportError as e:
        print(f"❌ Seaborn: {e}")

    try:
        import joblib
        print(f"✅ Joblib {joblib.__version__}")
    except ImportError as e:
        print(f"❌ Joblib: {e}")

    try:
        import mediapipe as mp
        print("✅ MediaPipe imported successfully")
    except ImportError as e:
        print(f"❌ MediaPipe: {e}")

    try:
        import cv2
        print(f"✅ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV: {e}")

if __name__ == "__main__":
    test_imports()
