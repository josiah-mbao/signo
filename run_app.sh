#!/bin/bash

# Signo ASL Gesture Recognition - Single Command Runner
# This script automatically starts the Streamlit app and ngrok tunnel

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting Signo ASL Gesture Recognition App${NC}"
echo -e "${BLUE}================================================${NC}"

# Check if ngrok is installed
if ! command -v ngrok &> /dev/null && ! command -v ./ngrok &> /dev/null; then
    echo -e "${RED}❌ ngrok not found. Installing...${NC}"

    # Download ngrok for macOS ARM64
    echo -e "${YELLOW}📥 Downloading ngrok...${NC}"
    curl -L https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-darwin-arm64.zip -o ngrok.zip
    unzip ngrok.zip
    chmod +x ngrok

    echo -e "${GREEN}✅ ngrok installed${NC}"
else
    echo -e "${GREEN}✅ ngrok already available${NC}"
fi

# Check if Python dependencies are installed
echo -e "${YELLOW}🔍 Checking Python dependencies...${NC}"
if ! python -c "import streamlit, cv2, mediapipe, numpy" &> /dev/null; then
    echo -e "${YELLOW}📦 Installing Python dependencies...${NC}"
    pip install -r requirements.txt
    echo -e "${GREEN}✅ Dependencies installed${NC}"
else
    echo -e "${GREEN}✅ Dependencies already installed${NC}"
fi

# Start Streamlit app in background
echo -e "${GREEN}🎯 Starting Streamlit app on port 8503...${NC}"
streamlit run final.py --server.port 8503 --server.address 0.0.0.0 &
STREAMLIT_PID=$!

# Wait for Streamlit to start
echo -e "${YELLOW}⏳ Waiting for Streamlit to initialize...${NC}"
sleep 5

# Check if Streamlit is running
if kill -0 $STREAMLIT_PID 2>/dev/null; then
    echo -e "${GREEN}✅ Streamlit app started successfully${NC}"
else
    echo -e "${RED}❌ Failed to start Streamlit app${NC}"
    exit 1
fi

# Start ngrok tunnel
echo -e "${GREEN}🌐 Starting ngrok tunnel...${NC}"
./ngrok http 8503 &
NGROK_PID=$!

# Wait for ngrok to establish tunnel
echo -e "${YELLOW}⏳ Waiting for ngrok tunnel to establish...${NC}"
sleep 8

# Get ngrok public URL
NGROK_URL=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null | python -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'tunnels' in data and len(data['tunnels']) > 0:
        print(data['tunnels'][0]['public_url'])
except:
    pass
")

if [ -n "$NGROK_URL" ]; then
    echo -e "${GREEN}✅ App is live and accessible at:${NC}"
    echo -e "${BLUE}🌍 $NGROK_URL${NC}"
    echo ""
    echo -e "${YELLOW}📱 Share this URL with others to access the ASL gesture recognition app!${NC}"
    echo -e "${YELLOW}💡 The app includes:${NC}"
    echo -e "${YELLOW}   • Real-time gesture recognition using webcam${NC}"
    echo -e "${YELLOW}   • Sentence building with gesture controls${NC}"
    echo -e "${YELLOW}   • Support for letters and phrases${NC}"
    echo ""
    echo -e "${BLUE}Press Ctrl+C to stop the application${NC}"
else
    echo -e "${RED}❌ Failed to get ngrok URL. Check ngrok status at http://localhost:4040${NC}"
    echo -e "${YELLOW}💡 You can still access locally at: http://localhost:8503${NC}"
fi

# Wait for user to stop (Ctrl+C)
trap "echo -e '\n${YELLOW}🛑 Stopping services...${NC}'; kill $STREAMLIT_PID $NGROK_PID 2>/dev/null; echo -e '${GREEN}✅ Services stopped${NC}'; exit" INT
wait
