import streamlit as st
import cv2
import torch
import numpy as np
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import pandas as pd
import sys
import os
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(PROJECT_DIR)

from model import ModelBuilder

class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = r"checkpoints\best_model.pth"
    EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    
    MEAN = torch.tensor([0.485, 0.485, 0.485]).view(3, 1, 1)
    STD = torch.tensor([0.229, 0.229, 0.229]).view(3, 1, 1)
    
    EMOTION_COLORS = {
        "Angry": (0, 0, 255),
        "Disgust": (128, 0, 128),
        "Fear": (128, 128, 0),
        "Happy": (0, 255, 0),
        "Sad": (255, 0, 0),
        "Surprise": (0, 255, 255),
        "Neutral": (192, 192, 192)
    }
    
    EMOTION_EMOJIS = {
        "Angry": "😠",
        "Disgust": "🤢",
        "Fear": "😨",
        "Happy": "😊",
        "Sad": "😢",
        "Surprise": "😲",
        "Neutral": "😐"
    }
    
    RTC_CONFIGURATION = {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
        ]
    }
    
    CAMERA_SETTINGS = {
        "video": {
            "width": {"ideal": 640},
            "height": {"ideal": 480},
            "frameRate": {"ideal": 15}
        },
        "audio": False
    }

torch.set_grad_enabled(False)

@st.cache_resource
def load_model():
    try:
        model = ModelBuilder(
            num_classes=7,
            model_name="resnet18",
            pretrained=True,
            freeze_backbone=False,
            print_summary=False
        ).to(Config.DEVICE)
        
        if not os.path.exists(Config.MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {Config.MODEL_PATH}")
        
        checkpoint = torch.load(Config.MODEL_PATH, map_location=Config.DEVICE)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def preprocess_frame(frame):
    resized = cv2.resize(frame, (224, 224))
    
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    img = rgb.astype(np.float32) / 255.0

    img = np.transpose(img, (2, 0, 1)).copy()
    
    img = torch.tensor(img, dtype=torch.float32)
    
    img = (img - Config.MEAN) / Config.STD
    
    return img.unsqueeze(0).to(Config.DEVICE)

class EmotionVideoProcessor(VideoProcessorBase):   
    def __init__(self):
        self.model = load_model()
        self.emotion = "Initializing..."
        self.confidence = 0.0
        self.fps = 0
        self.last_time = time.time()
        self.frame_count = 0
        self.probabilities = [0.0] * 7
        self.error_count = 0
        self.max_errors = 5
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_time = current_time
        
        if self.model is None:
            self._draw_error(img, "Model not loaded")
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        try:
            input_tensor = preprocess_frame(img)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probs, 1)
                
                self.emotion = Config.EMOTIONS[predicted.item()]
                self.confidence = confidence.item() * 100
                self.probabilities = probs[0].cpu().numpy()
            
            self._draw_results(img)
            
            self.error_count = 0
            
        except Exception as e:
            self.error_count += 1
            if self.error_count <= self.max_errors:
                self._draw_error(img, f"Error: {str(e)[:30]}")
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    def _draw_results(self, img):
        color = Config.EMOTION_COLORS.get(self.emotion, (255, 255, 255))
        
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (450, 120), (20, 20, 30), -1)
        img[:] = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
        
        cv2.rectangle(img, (10, 10), (450, 120), color, 3)
        
        cv2.putText(img, f"Emotion: {self.emotion}", (25, 45),
                   cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2, cv2.LINE_AA)
        cv2.putText(img, f"Confidence: {self.confidence:.1f}%", (25, 75),
                   cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"FPS: {self.fps}", (25, 105),
                   cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    
    def _draw_error(self, img, error_msg):
        cv2.putText(img, error_msg, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def inject_custom_css():
    st.markdown("""
        <style>
        /* Custom animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
        }
        
        @keyframes slideInLeft {
            from { opacity: 0; transform: translateX(-30px); }
            to { opacity: 1; transform: translateX(0); }
        }
        
        @keyframes zoomIn {
            from { transform: scale(0.9); opacity: 0; }
            to { transform: scale(1); opacity: 1; }
        }
        
        /* Custom classes */
        .app-header {
            text-align: center;
            margin-bottom: 2rem;
            animation: fadeIn 1.2s ease-out;
        }
        
        .app-title {
            font-size: 2.5rem;
            font-weight: 900;
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        
        .app-subtitle {
            font-size: 1.1rem;
            color: #64748b;
            margin-bottom: 1rem;
        }
        
        .badge {
            display: inline-block;
            padding: 0.4rem 1rem;
            background: rgba(59, 130, 246, 0.1);
            border: 1px solid rgba(59, 130, 246, 0.3);
            border-radius: 50px;
            font-size: 0.9rem;
            color: #3b82f6;
            animation: float 3s ease-in-out infinite;
        }
        
        .glass-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 1.5rem;
            margin: 1rem 0;
            animation: zoomIn 0.8s ease-out;
        }
        
        .feature-card {
            background: rgba(59, 130, 246, 0.05);
            border: 1px solid rgba(59, 130, 246, 0.2);
            border-radius: 12px;
            padding: 1.2rem;
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(59, 130, 246, 0.3);
        }
        
        .feature-icon {
            font-size: 2.5rem;
            animation: float 3s ease-in-out infinite;
        }
        
        .feature-title {
            font-size: 0.9rem;
            font-weight: 600;
            margin-top: 0.5rem;
        }
        
        .section-title {
            font-size: 1.5rem;
            font-weight: 800;
            margin: 2rem 0 1rem;
            animation: slideInLeft 1s ease-out;
        }
        
        /* Metric styling */
        .stMetric {
            background: rgba(59, 130, 246, 0.05);
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid rgba(59, 130, 246, 0.1);
        }
        </style>
    """, unsafe_allow_html=True)

def render_header():
    st.markdown("""
        <div class='app-header'>
            <h1 class='app-title'>📷 Live Webcam Emotion Detection</h1>
            <p class='app-subtitle'>
                Real-time facial emotion recognition powered by ResNet18
            </p>
            <span class='badge'>✨ Live AI Detection</span>
        </div>
    """, unsafe_allow_html=True)

def render_model_status(model):
    if model is None:
        st.error("❌ Failed to load model. Please check the model path.")
        st.info(f"Expected model at: `{Config.MODEL_PATH}`")
        st.stop()
        return False
    
    col1, col2 = st.columns(2)
    with col1:
        st.success("✅ Model loaded successfully")
    with col2:
        st.info(f"🖥 Running on {Config.DEVICE.upper()}")
    
    return True

def render_instructions():
    st.markdown("<h3 class='section-title'>📋 How to Use</h3>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class='glass-card'>
            <ol style='line-height: 2; font-size: 15px;'>
                <li>🔐 <strong>Allow camera access</strong> when prompted by your browser</li>
                <li>🎥 <strong>Position your face</strong> clearly in front of the camera</li>
                <li>😊 <strong>Express different emotions</strong> and watch the AI detect them instantly</li>
                <li>📊 <strong>Monitor statistics</strong> below the video feed in real-time</li>
                <li>⏹ <strong>Click "STOP"</strong> button when finished</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)

def render_settings():
    st.markdown("<h3 class='section-title'>⚙ Display Settings</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        show_emotions_list = st.checkbox("🎭 Emotion Classes", value=True)
    with col2:
        pass
    
    return show_emotions_list

def render_webcam_stream():
    st.markdown("<h3 class='section-title'>🎥 Live Camera Feed</h3>", unsafe_allow_html=True)
    
    rtc_configuration = RTCConfiguration(Config.RTC_CONFIGURATION)
    
    webrtc_ctx = webrtc_streamer(
        key="emotion-detection",
        video_processor_factory=EmotionVideoProcessor,
        rtc_configuration=rtc_configuration,
        media_stream_constraints=Config.CAMERA_SETTINGS,
        async_processing=True,
    )
    
    return webrtc_ctx

def render_live_stats(webrtc_ctx, show_probabilities):
    if not webrtc_ctx.video_processor:
        st.info("📹 Start the camera to see live statistics")
        return
    
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 class='section-title'>📊 Real-time Statistics</h3>", unsafe_allow_html=True)
    
    # Create placeholders for metrics
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    emotion_placeholder = stats_col1.empty()
    confidence_placeholder = stats_col2.empty()
    fps_placeholder = stats_col3.empty()
    emoji_placeholder = stats_col4.empty()
    
    prob_placeholder = st.empty() if show_probabilities else None

    # Get the processor
    processor = webrtc_ctx.video_processor
    
    # Update metrics - these will update automatically when the processor updates
    emotion_placeholder.metric(
        "Current Emotion",
        processor.emotion,
        delta=f"{processor.confidence:.1f}%"
    )
    
    confidence_placeholder.metric(
        "Confidence",
        f"{processor.confidence:.1f}%"
    )
    
    fps_placeholder.metric(
        "FPS",
        processor.fps
    )
    
    emoji = Config.EMOTION_EMOJIS.get(processor.emotion, "😐")
    emoji_placeholder.markdown(f"""
        <div style='text-align: center; font-size: 48px; animation: float 3s ease-in-out infinite;'>
            {emoji}
        </div>
    """, unsafe_allow_html=True)
    
    # Show probabilities table if enabled
    if show_probabilities and prob_placeholder:
        prob_data = {
            "Emotion": Config.EMOTIONS,
            "Probability": [f"{p*100:.1f}%" for p in processor.probabilities]
        }
        df = pd.DataFrame(prob_data)
        prob_placeholder.dataframe(df, use_container_width=True, hide_index=True)

def render_emotion_classes():
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 class='section-title'>🎭 Detectable Emotions</h3>", unsafe_allow_html=True)
    
    cols = st.columns(7)
    
    for idx, (col, emotion) in enumerate(zip(cols, Config.EMOTIONS)):
        with col:
            emoji = Config.EMOTION_EMOJIS.get(emotion, "😐")
            st.markdown(f"""
                <div class='feature-card'>
                    <div class='feature-icon'>{emoji}</div>
                    <div class='feature-title'>{emotion}</div>
                </div>
            """, unsafe_allow_html=True)

def render_tips():
    st.markdown("<hr>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class='glass-card'>
                <h4>💡 Tips for Best Results</h4>
                <ul style='line-height: 2; font-size: 14px;'>
                    <li>✨ Ensure <strong>good lighting</strong> conditions</li>
                    <li>👤 Keep your <strong>face clearly visible</strong></li>
                    <li>📐 Avoid <strong>extreme angles</strong></li>
                    <li>😊 Try <strong>different expressions</strong> for testing</li>
                    <li>🎯 Stay <strong>centered</strong> in the frame</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='glass-card'>
                <h4>⚡ Performance Tips</h4>
                <ul style='line-height: 2; font-size: 14px;'>
                    <li>🖥 Close <strong>unnecessary browser tabs</strong></li>
                    <li>🚀 Use <strong>GPU</strong> if available</li>
                    <li>🌐 Ensure <strong>stable internet</strong> connection</li>
                    <li>📹 Use <strong>good quality</strong> webcam</li>
                    <li>🔄 Refresh page if <strong>FPS drops</strong></li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with st.expander("🔧 Troubleshooting Guide", expanded=False):
        st.markdown("""
            **📹 Camera not working?**
            - Check browser camera permissions in settings
            - Try refreshing the page (F5)
            - Ensure no other app is using the camera
            - Try a different browser (Chrome recommended)
            
            **⚡ Low FPS or lag?**
            - Close other browser tabs and applications
            - Ensure GPU is available and enabled
            - Lower camera resolution in browser settings
            - Check your internet connection speed
            
            **🎯 Inaccurate predictions?**
            - Improve lighting conditions (natural light is best)
            - Face the camera directly, not at an angle
            - Ensure your entire face is visible
            - Try exaggerating facial expressions
        """)

def render_footer():
    st.markdown("""
        <div style='text-align: center; margin-top: 3rem; padding: 2rem; 
                    color: #64748b; font-size: 13px;'>
            <p>🎭 Powered by <strong>ResNet18</strong> & <strong>PyTorch</strong></p>
            <p style='margin-top: 0.5rem;'>Built with ❤ by THE BRO Team</p>
        </div>
    """, unsafe_allow_html=True)

def webcam_page():
    
    inject_custom_css()
    
    render_header()
    
    with st.spinner("🔄 Loading AI Model..."):
        model = load_model()
    
    if not render_model_status(model):
        return
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    render_instructions()
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    show_emotions_list = render_settings()
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    webrtc_ctx = render_webcam_stream()
    
    if show_emotions_list:
        render_emotion_classes()
    
    render_tips()
    
    render_footer()

if __name__ == "__main__":
    webcam_page()
