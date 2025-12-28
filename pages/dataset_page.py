import streamlit as st
from PIL import Image
import random
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from preprocessingg import FERDatasetLoader

def dataset_page():
    st.markdown("""
        <div class='app-header' style='animation: fadeIn 1.2s ease-out;'>
            <h1 class='app-title' style='animation: fadeIn 1.4s ease-out;'>📚 Dataset Explorer</h1>
            <p class='app-subtitle' style='animation: fadeIn 1.6s ease-out;'>
                Explore FER-2013 Dataset Structure & Preprocessing Pipeline
            </p>
            <span class='badge' style='animation: float 3s ease-in-out infinite;'>✨ 35,887 Images</span>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    train_dir = r"fer2013\train"
    test_dir = r"fer2013\test"
    
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        st.error("❌ Dataset folders not found! Please check the paths:")
        st.code(f"Train: {train_dir}\nTest: {test_dir}")
        st.info("💡 Make sure you have extracted the FER-2013 dataset in the correct location.")
        st.stop()
        return
    
    st.markdown("<h3 class='section-title' style='animation: slideInLeft 1s ease-out;'>📊 Dataset Summary</h3>", unsafe_allow_html=True)
    
    with st.spinner("🔄 Loading dataset..."):
        try:
            loader = FERDatasetLoader(
                train_dir=train_dir, 
                test_dir=test_dir,
                batch_size=32,
                num_workers=0 
            )
            
            train_loader, val_loader, test_loader, total_samples = loader.get_loaders()
            lengths = loader.get_len()
            classes = loader.get_classes()
            
        except Exception as e:
            st.error(f"❌ Error loading dataset: {str(e)}")
            st.stop()
            return
    
    st.success("✅ Dataset loaded successfully!")
    
    st.markdown("""
        <div class='glass-card' style='animation: zoomIn 0.8s ease-out;'>
            <h4 style='color: var(--accent-primary); margin-top: 0;'>📈 Dataset Statistics</h4>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🎓 Training Samples", 
            f"{lengths['train_samples']:,}",
            delta="80% of train set"
        )
    
    with col2:
        st.metric(
            "✅ Validation Samples", 
            f"{lengths['val_samples']:,}",
            delta="20% of train set"
        )
    
    with col3:
        st.metric(
            "🧪 Test Samples", 
            f"{lengths['test_samples']:,}",
            delta="Separate test set"
        )
    
    with col4:
        st.metric(
            "🎭 Emotion Classes", 
            len(classes)
        )
    
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 class='section-title' style='animation: slideInLeft 1s ease-out;'>🎭 Emotion Classes</h3>", unsafe_allow_html=True)
    
    emotion_emojis = {
        "angry": "😠",
        "disgust": "🤢",
        "fear": "😨",
        "happy": "😊",
        "sad": "😢",
        "surprise": "😲",
        "neutral": "😐"
    }
    
    cols = st.columns(7)
    for idx, (col, class_name) in enumerate(zip(cols, classes)):
        with col:
            emoji = emotion_emojis.get(class_name.lower(), "😐")
            
            class_folder = os.path.join(train_dir, class_name)
            if os.path.exists(class_folder):
                num_images = len(os.listdir(class_folder))
            else:
                num_images = 0
            
            st.markdown(f"""
                <div class='feature-card' style='padding: 1.2rem; animation: fadeIn {1 + idx*0.1}s ease-out;'>
                    <div class='feature-icon' style='font-size: 36px; animation: float {3 + idx*0.2}s ease-in-out infinite;'>{emoji}</div>
                    <div class='feature-title' style='font-size: 14px; margin-top: 0.5rem;'>{class_name.title()}</div>
                    <div class='feature-desc' style='font-size: 12px; margin-top: 0.3rem;'>{num_images:,} images</div>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 class='section-title' style='animation: slideInLeft 1s ease-out;'>🖼️ Sample Images from Dataset</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_class = st.selectbox(
            "Select an emotion class:",
            classes,
            index=3  
        )
        
        num_samples = st.slider(
            "Number of samples to display:",
            min_value=1,
            max_value=12,
            value=6
        )
    
    with col2:
        class_folder = os.path.join(train_dir, selected_class)
        
        if os.path.exists(class_folder):
            image_files = os.listdir(class_folder)
            
            if len(image_files) > 0:
                sample_files = random.sample(image_files, min(num_samples, len(image_files)))
                
                cols_per_row = 3
                rows = (len(sample_files) + cols_per_row - 1) // cols_per_row
                
                for row in range(rows):
                    cols = st.columns(cols_per_row)
                    for col_idx in range(cols_per_row):
                        img_idx = row * cols_per_row + col_idx
                        if img_idx < len(sample_files):
                            img_path = os.path.join(class_folder, sample_files[img_idx])
                            img = Image.open(img_path)
                            
                            with cols[col_idx]:
                                st.image(
                                    img, 
                                    caption=f"{selected_class} - {img_idx+1}",
                                    use_container_width=True
                                )
            else:
                st.warning("⚠️ No images found in this class folder!")
        else:
            st.error("❌ Class folder not found!")
    
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 class='section-title' style='animation: slideInLeft 1s ease-out;'>🔧 Preprocessing Pipeline</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class='glass-card' style='animation: zoomIn 0.8s ease-out;'>
                <h4 style='color: var(--accent-primary); margin-top: 0;'>🎓 Training Transformations</h4>
                <ul style='line-height: 2; color: var(--text-secondary); font-size: 14px;'>
                    <li>📐 <strong>Resize:</strong> 224×224 pixels</li>
                    <li>⚫ <strong>Grayscale to RGB:</strong> 1 → 3 channels</li>
                    <li>🔄 <strong>Random Horizontal Flip:</strong> 50% chance</li>
                    <li>🔁 <strong>Random Rotation:</strong> ±10 degrees</li>
                    <li>🎨 <strong>Color Jitter:</strong> Brightness & Contrast ±20%</li>
                    <li>📊 <strong>Normalization:</strong> mean=0.485, std=0.229</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='glass-card' style='animation: zoomIn 0.8s ease-out;'>
                <h4 style='color: var(--accent-secondary); margin-top: 0;'>✅ Validation/Test Transformations</h4>
                <ul style='line-height: 2; color: var(--text-secondary); font-size: 14px;'>
                    <li>📐 <strong>Resize:</strong> 224×224 pixels</li>
                    <li>⚫ <strong>Grayscale to RGB:</strong> 1 → 3 channels</li>
                    <li>📊 <strong>Normalization:</strong> mean=0.485, std=0.229</li>
                    <li>❌ <strong>No Augmentation:</strong> Pure evaluation</li>
                </ul>
                <br>
                <p style='color: var(--text-secondary); font-size: 13px; margin-top: 1rem;'>
                    💡 <em>Validation and test sets use simpler transformations to evaluate true model performance.</em>
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 class='section-title' style='animation: slideInLeft 1s ease-out;'>👁️ Preprocessing Visualization</h3>", unsafe_allow_html=True)
    
    if st.button("🎲 Generate Random Sample with Transformations", use_container_width=True):
        random_class = random.choice(classes)
        class_folder = os.path.join(train_dir, random_class)
        image_files = os.listdir(class_folder)
        random_image_path = os.path.join(class_folder, random.choice(image_files))
        
        original_img = Image.open(random_image_path)
        
        st.markdown(f"""
            <div class='glass-card' style='animation: zoomIn 0.8s ease-out; text-align: center;'>
                <h4 style='color: var(--accent-primary);'>Selected: {random_class.title()}</h4>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("**1️⃣ Original (48×48)**")
            st.image(original_img, use_container_width=True)
        
        with col2:
            st.markdown("**2️⃣ Resized (224×224)**")
            resized = original_img.resize((224, 224))
            st.image(resized, use_container_width=True)
        
        with col3:
            st.markdown("**3️⃣ Grayscale → RGB**")
            gray = resized.convert('L')
            rgb = Image.merge('RGB', (gray, gray, gray))
            st.image(rgb, use_container_width=True)
        
        with col4:
            st.markdown("**4️⃣ Normalized Tensor**")
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485]*3, std=[0.229]*3)
            ])
            normalized = transform(rgb)
            
            display_img = normalized * 0.229 + 0.485
            display_img = display_img.permute(1, 2, 0).numpy()
            display_img = np.clip(display_img, 0, 1)
            
            st.image(display_img, use_container_width=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 class='section-title' style='animation: slideInLeft 1s ease-out;'>⚙️ Technical Details</h3>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class='glass-card' style='animation: zoomIn 0.8s ease-out;'>
            <h4 style='color: var(--accent-primary); margin-top: 0;'>📋 DataLoader Configuration</h4>
            <ul style='line-height: 2; color: var(--text-secondary); font-size: 14px;'>
                <li>📦 <strong>Batch Size:</strong> 32 samples per batch</li>
                <li>🔀 <strong>Training Shuffle:</strong> Enabled (for better generalization)</li>
                <li>⚡ <strong>Number of Workers:</strong> 4 parallel data loading threads</li>
                <li>📌 <strong>Pin Memory:</strong> Enabled (faster GPU transfer)</li>
                <li>🎲 <strong>Random Seed:</strong> 42 (for reproducibility)</li>
                <li>📊 <strong>Train/Val Split:</strong> 80% / 20%</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    with st.expander("📊 Detailed Dataset Splits Information"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🎓 Training Set**")
            st.write(f"- Samples: {lengths['train_samples']:,}")
            st.write(f"- Batches: {lengths['train_batches']}")
            st.write(f"- Augmentation: ✅ Enabled")
            st.write(f"- Shuffle: ✅ Yes")
        
        with col2:
            st.markdown("**✅ Validation Set**")
            st.write(f"- Samples: {lengths['val_samples']:,}")
            st.write(f"- Batches: {lengths['val_batches']}")
            st.write(f"- Augmentation: ❌ Disabled")
            st.write(f"- Shuffle: ❌ No")
        
        with col3:
            st.markdown("**🧪 Test Set**")
            st.write(f"- Samples: {lengths['test_samples']:,}")
            st.write(f"- Batches: {lengths['test_batches']}")
            st.write(f"- Augmentation: ❌ Disabled")
            st.write(f"- Shuffle: ❌ No")
    
    st.markdown("""
        <div style='text-align: center; margin-top: 3rem; padding: 2rem; 
                    color: var(--text-secondary); font-size: 13px;'>
            <p>📚 FER-2013 Dataset | Preprocessed for Deep Learning</p>
            <p style='margin-top: 0.5rem;'>Built with ❤️ by THE BRO Team</p>
        </div>
    """, unsafe_allow_html=True)