#!/bin/bash

# Signo ASL Gesture Recognition - Local Runner
# This script automatically starts the Streamlit app locally

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting Signo ASL Gesture Recognition App${NC}"
echo -e "${BLUE}================================================${NC}"

# Check if Python dependencies are installed
echo -e "${YELLOW}🔍 Checking Python dependencies...${NC}"
if ! python -c "import streamlit, cv2, mediapipe, numpy" &> /dev/null; then
    echo -e "${YELLOW}📦 Installing Python dependencies...${NC}"
    pip install -r requirements.txt
    echo -e "${GREEN}✅ Dependencies installed${NC}"
else
    echo -e "${GREEN}✅ Dependencies already installed${NC}"
fi

# Start Streamlit app
echo -e "${GREEN}🎯 Starting Streamlit app on port 8503...${NC}"
echo -e "${YELLOW}📱 Access the app at: http://localhost:8503${NC}"
echo ""
echo -e "${YELLOW}💡 The app includes:${NC}"
echo -e "${YELLOW}   • Real-time gesture recognition using webcam${NC}"
echo -e "${YELLOW}   • Sentence building with gesture controls${NC}"
echo -e "${YELLOW}   • Support for letters and phrases${NC}"
echo ""
echo -e "${BLUE}Press Ctrl+C to stop the application${NC}"
echo ""

# Run Streamlit app (this will run in foreground)
streamlit run final.py --server.port 8503 --server.address 0.0.0.0
