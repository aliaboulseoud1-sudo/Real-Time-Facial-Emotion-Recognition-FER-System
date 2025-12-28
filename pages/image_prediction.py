import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import torch
import os
import sys
import numpy as np
import zipfile
import tempfile
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from predict import EmotionPredictor
from config import Config

def image_prediction_page():
    
    st.markdown("""
        <div class='app-header' style='animation: fadeIn 1.2s ease-out;'>
            <h1 class='app-title' style='animation: fadeIn 1.4s ease-out;'>📤 Upload Image Prediction</h1>
            <p class='app-subtitle' style='animation: fadeIn 1.6s ease-out;'>
                Upload single or multiple images for emotion classification
            </p>
            <span class='badge' style='animation: float 3s ease-in-out infinite;'>✨ AI-Powered Analysis</span>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    st.markdown("<h3 class='section-title' style='animation: slideInLeft 1s ease-out;'>⚙️ Model Configuration</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_option = st.selectbox(
            "🧠 Select Model Architecture:",
            ["resnet18", "mobilenetv2"],
            index=0
        )
    
    with col2:
        MODEL_PATH = st.text_input(
            "📁 Model Checkpoint Path:",
            Config.BEST_MODEL_PATH
        )
    
    @st.cache_resource
    def load_predictor(model_path, model_name):
        try:
            Config.MODEL_NAME = model_name
            
            predictor = EmotionPredictor(
                checkpoint_path=model_path,
                config=Config
            )
            return predictor, None
        except Exception as e:
            return None, str(e)
    
    with st.spinner("🔄 Loading model..."):
        predictor, error = load_predictor(MODEL_PATH, model_option)
    
    if error:
        st.error(f"❌ Error loading model: {error}")
        st.info("💡 Please check the model path and try again.")
        st.stop()
        return
    
    if predictor:
        st.success(f"✅ Model loaded successfully! Running on **{str(Config.DEVICE).upper()}**")
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📷 Single Image", "📁 Multiple Images (ZIP)"])
    
    with tab1:
        st.markdown("<h3 class='section-title' style='animation: slideInLeft 1s ease-out;'>📷 Upload Single Image</h3>", unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["png", "jpg", "jpeg", "bmp"],
            key="single_image",
            help="Upload a facial image for emotion detection"
        )
        
        if uploaded_file:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("""
                    <div class='glass-card' style='animation: zoomIn 0.8s ease-out;'>
                        <h4 style='color: var(--accent-primary); margin-top: 0;'>📸 Uploaded Image</h4>
                    </div>
                """, unsafe_allow_html=True)
                
                img = Image.open(uploaded_file)
                st.image(img, use_container_width=True)
            
            with col2:
                st.markdown("""
                    <div class='glass-card' style='animation: zoomIn 0.8s ease-out;'>
                        <h4 style='color: var(--accent-secondary); margin-top: 0;'>🎯 Prediction Results</h4>
                    </div>
                """, unsafe_allow_html=True)
                
                with st.spinner("🔄 Analyzing image..."):
                    uploaded_file.seek(0)
                    
                    try:
                        label, conf, probs = predictor.predict_image(
                            uploaded_file, 
                            return_probabilities=True
                        )
                        
                        emotion_emojis = {
                            "angry": "😠",
                            "disgust": "🤢",
                            "fear": "😨",
                            "happy": "😊",
                            "sad": "😢",
                            "surprise": "😲",
                            "neutral": "😐"
                        }
                        
                        emoji = emotion_emojis.get(label.lower(), "😐")
                        
                        st.markdown(f"""
                            <div style='text-align: center; padding: 2rem; 
                                        background: var(--glass-bg); backdrop-filter: blur(10px);
                                        border: 2px solid var(--accent-primary); border-radius: 16px;
                                        animation: zoomIn 0.6s ease-out;'>
                                <div style='font-size: 64px; animation: float 3s ease-in-out infinite;'>{emoji}</div>
                                <h2 style='color: var(--accent-primary); margin: 1rem 0;'>{label.title()}</h2>
                                <p style='font-size: 24px; color: var(--text-primary); font-weight: 600;'>
                                    {conf*100:.2f}% Confidence
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        st.markdown("**📊 All Emotion Probabilities:**")
                        
                        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                        
                        for emotion, prob in sorted_probs:
                            emoji = emotion_emojis.get(emotion.lower(), "😐")
                            
                            progress_color = "var(--accent-primary)" if emotion == label else "var(--text-secondary)"
                            
                            st.markdown(f"""
                                <div style='margin: 0.5rem 0;'>
                                    <div style='display: flex; justify-content: space-between; margin-bottom: 0.3rem;'>
                                        <span style='font-weight: 600;'>{emoji} {emotion.title()}</span>
                                        <span style='color: {progress_color}; font-weight: 600;'>{prob*100:.1f}%</span>
                                    </div>
                                    <div style='background: var(--bg-secondary); border-radius: 10px; height: 8px; overflow: hidden;'>
                                        <div style='background: {progress_color}; width: {prob*100}%; height: 100%; 
                                                    border-radius: 10px; transition: width 0.5s ease;'></div>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"❌ Error during prediction: {str(e)}")
            
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("<h3 class='section-title' style='animation: slideInLeft 1s ease-out;'>📊 Detailed Visualization</h3>", unsafe_allow_html=True)
            
            if st.button("🎨 Generate Visualization", use_container_width=True):
                with st.spinner("Creating visualization..."):
                    uploaded_file.seek(0)
                    fig = predictor.visualize_prediction(
                        uploaded_file,
                        show=False
                    )
                    st.pyplot(fig)
                    plt.close()
    
    with tab2:
        st.markdown("<h3 class='section-title' style='animation: slideInLeft 1s ease-out;'>📁 Upload Multiple Images (ZIP)</h3>", unsafe_allow_html=True)
        
        st.markdown("""
            <div class='glass-card' style='animation: zoomIn 0.8s ease-out;'>
                <h4 style='color: var(--accent-primary); margin-top: 0;'>📋 Instructions</h4>
                <ol style='line-height: 2; color: var(--text-secondary);'>
                    <li>Create a ZIP file containing your images</li>
                    <li>Upload the ZIP file below</li>
                    <li>Wait for batch processing to complete</li>
                    <li>Download results as CSV/Excel</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)
        
        uploaded_zip = st.file_uploader(
            "Choose a ZIP file containing images",
            type=["zip"],
            key="zip_file",
            help="Upload a ZIP file with multiple facial images"
        )
        
        if uploaded_zip:
            with tempfile.TemporaryDirectory() as tmpdir:
                with st.spinner("📦 Extracting ZIP file..."):
                    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
                        zip_ref.extractall(tmpdir)
                
                image_files = []
                for root, dirs, files in os.walk(tmpdir):
                    for f in files:
                        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                            image_files.append(os.path.join(root, f))
                
                if not image_files:
                    st.warning("⚠️ No valid images found in the ZIP file!")
                else:
                    st.info(f"📸 Found {len(image_files)} images to process")
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    results = []
                    
                    for idx, img_path in enumerate(image_files):
                        status_text.text(f"Processing {idx+1}/{len(image_files)}: {os.path.basename(img_path)}")
                        
                        try:
                            label, conf = predictor.predict_image(img_path)
                            results.append({
                                "filename": os.path.basename(img_path),
                                "prediction": label,
                                "confidence": f"{conf*100:.2f}%"
                            })
                        except Exception as e:
                            results.append({
                                "filename": os.path.basename(img_path),
                                "prediction": "Error",
                                "confidence": str(e)
                            })
                        
                        progress_bar.progress((idx + 1) / len(image_files))
                    
                    status_text.empty()
                    progress_bar.empty()
                    
                    st.success(f"✅ Processing completed! Analyzed {len(results)} images")
                    
                    st.markdown("<hr>", unsafe_allow_html=True)
                    st.markdown("<h3 class='section-title' style='animation: slideInLeft 1s ease-out;'>📊 Results Summary</h3>", unsafe_allow_html=True)
                    
                    df = pd.DataFrame(results)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Images", len(results))
                    
                    with col2:
                        successful = len([r for r in results if r['prediction'] != 'Error'])
                        st.metric("Successful", successful)
                    
                    with col3:
                        failed = len([r for r in results if r['prediction'] == 'Error'])
                        st.metric("Failed", failed)
                    
                    emotion_dist = df[df['prediction'] != 'Error']['prediction'].value_counts()
                    
                    if len(emotion_dist) > 0:
                        st.markdown("**🎭 Emotion Distribution:**")
                        
                        for emotion, count in emotion_dist.items():
                            percentage = (count / len(results)) * 100
                            st.markdown(f"- **{emotion}**: {count} images ({percentage:.1f}%)")
                    
                    st.markdown("<hr>", unsafe_allow_html=True)
                    st.markdown("<h3 class='section-title' style='animation: slideInLeft 1s ease-out;'>📋 Detailed Results</h3>", unsafe_allow_html=True)
                    
                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    st.markdown("<hr>", unsafe_allow_html=True)
                    st.markdown("<h3 class='section-title' style='animation: slideInLeft 1s ease-out;'>💾 Download Results</h3>", unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download as CSV",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with col2:
                        try:
                            import io
                            buffer = io.BytesIO()
                            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                                df.to_excel(writer, index=False, sheet_name='Predictions')
                            
                            st.download_button(
                                label="📥 Download as Excel",
                                data=buffer.getvalue(),
                                file_name="predictions.xlsx",
                                mime="application/vnd.ms-excel",
                                use_container_width=True
                            )
                        except:
                            st.info("📦 Install xlsxwriter for Excel export: `pip install xlsxwriter`")
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    with st.expander("ℹ️ About the Model"):
        st.markdown(f"""
        **Model Information:**
        - Architecture: {model_option.upper()}
        - Input Size: {Config.IMAGE_SIZE}×{Config.IMAGE_SIZE}
        - Number of Classes: {Config.NUM_CLASSES}
        - Emotion Classes: {', '.join(map(str, Config.EMOTION_LABELS))}
        - Device: {str(Config.DEVICE).upper()}
        
        **Supported Formats:**
        - Single Image: PNG, JPG, JPEG, BMP
        - Batch Processing: ZIP file containing images
        """)
    
    st.markdown("""
        <div style='text-align: center; margin-top: 3rem; padding: 2rem; 
                    color: var(--text-secondary); font-size: 13px;'>
            <p>🎯 Accurate emotion detection powered by Deep Learning</p>
            <p style='margin-top: 0.5rem;'>Built with ❤️ by THE BRO Team</p>
        </div>
    """, unsafe_allow_html=True)