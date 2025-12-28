import streamlit as st
from PIL import Image
import os

st.set_page_config(
    page_title="Facial Emotion Recognition",
    page_icon="👁️‍🗨️",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True

def get_custom_css(dark_mode=True):
    if dark_mode:
        bg_primary = "#0e1117"
        bg_secondary = "#1a1d29"
        bg_card = "rgba(26, 29, 41, 0.7)"
        text_primary = "#ffffff"
        text_secondary = "#b4b9c5"
        accent_primary = "#3b82f6"
        accent_secondary = "#8b5cf6"
        border_color = "rgba(59, 130, 246, 0.2)"
        hover_glow = "rgba(59, 130, 246, 0.4)"
        glass_bg = "rgba(26, 29, 41, 0.6)"
    else:
        bg_primary = "#ffffff"
        bg_secondary = "#f8fafc"
        bg_card = "rgba(248, 250, 252, 0.9)"
        text_primary = "#1e293b"
        text_secondary = "#64748b"
        accent_primary = "#3b82f6"
        accent_secondary = "#8b5cf6"
        border_color = "rgba(59, 130, 246, 0.3)"
        hover_glow = "rgba(59, 130, 246, 0.3)"
        glass_bg = "rgba(255, 255, 255, 0.7)"

    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

    /* =============== ROOT VARIABLES =============== */
    :root {{
        --bg-primary: {bg_primary};
        --bg-secondary: {bg_secondary};
        --bg-card: {bg_card};
        --text-primary: {text_primary};
        --text-secondary: {text_secondary};
        --accent-primary: {accent_primary};
        --accent-secondary: {accent_secondary};
        --border-color: {border_color};
        --hover-glow: {hover_glow};
        --glass-bg: {glass_bg};
    }}

    /* =============== ANIMATIONS =============== */
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    @keyframes slideInLeft {{
        from {{ opacity: 0; transform: translateX(-30px); }}
        to {{ opacity: 1; transform: translateX(0); }}
    }}

    @keyframes zoomIn {{
        from {{ transform: scale(0.9); opacity: 0; }}
        to {{ transform: scale(1); opacity: 1; }}
    }}

    @keyframes glow {{
        0%, 100% {{ box-shadow: 0 0 20px var(--hover-glow); }}
        50% {{ box-shadow: 0 0 40px var(--hover-glow), 0 0 60px var(--hover-glow); }}
    }}

    @keyframes gradientShift {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}

    @keyframes float {{
        0%, 100% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-10px); }}
    }}

    /* =============== PARTICLES BACKGROUND =============== */
    .particles-bg {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: -1;
        opacity: 0.15;
        background: 
            radial-gradient(circle at 20% 30%, var(--accent-primary) 0%, transparent 50%),
            radial-gradient(circle at 80% 70%, var(--accent-secondary) 0%, transparent 50%);
        animation: gradientShift 15s ease infinite;
        background-size: 200% 200%;
        pointer-events: none;
    }}

    /* =============== GLOBAL STYLES =============== */
    * {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    .stApp {{
        background: var(--bg-primary);
        color: var(--text-primary);
    }}

    .block-container {{
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1400px;
        animation: fadeIn 1s ease-out;
    }}

    /* =============== HEADER SECTION =============== */
    .app-header {{
        text-align: center;
        margin-bottom: 3rem;
        animation: fadeIn 1.2s ease-out;
    }}

    .app-title {{
        font-size: clamp(32px, 5vw, 56px);
        font-weight: 900;
        background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
        animation: fadeIn 1.4s ease-out;
        line-height: 1.2;
    }}

    .app-subtitle {{
        font-size: clamp(14px, 2vw, 18px);
        color: var(--text-secondary);
        font-weight: 400;
        margin-bottom: 1rem;
        animation: fadeIn 1.6s ease-out;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    }}

    .badge {{
        display: inline-block;
        padding: 0.4rem 1rem;
        background: {glass_bg};
        backdrop-filter: blur(10px);
        border: 1px solid var(--border-color);
        border-radius: 50px;
        font-size: 13px;
        font-weight: 600;
        color: var(--accent-primary);
        margin-top: 1rem;
        animation: float 3s ease-in-out infinite;
    }}

    /* =============== GLASS CARDS =============== */
    .glass-card {{
        background: {glass_bg};
        backdrop-filter: blur(12px);
        border: 1px solid var(--border-color);
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        animation: zoomIn 0.8s ease-out;
    }}

    .glass-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 20px 60px var(--hover-glow);
        border-color: var(--accent-primary);
    }}

    .feature-card {{
        background: {glass_bg};
        backdrop-filter: blur(10px);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
        transition: all 0.3s ease;
        cursor: pointer;
        height: 100%;
        animation: fadeIn 1s ease-out;
    }}

    .feature-card:hover {{
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 15px 40px var(--hover-glow);
        border-color: var(--accent-primary);
        background: {glass_bg};
    }}

    .feature-icon {{
        font-size: 48px;
        margin-bottom: 1rem;
        display: inline-block;
        animation: float 3s ease-in-out infinite;
    }}

    .feature-title {{
        font-size: 18px;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
    }}

    .feature-desc {{
        font-size: 14px;
        color: var(--text-secondary);
        line-height: 1.6;
    }}

    /* =============== IMAGE STYLING =============== */
    img {{
        border-radius: 16px;
        transition: all 0.4s ease;
        animation: zoomIn 1.2s ease-out;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
    }}

    img:hover {{
        transform: scale(1.03);
        box-shadow: 0 20px 60px rgba(59, 130, 246, 0.3);
    }}

    /* =============== SIDEBAR =============== */
    [data-testid="stSidebar"] {{
        background: var(--bg-secondary);
        border-right: 1px solid var(--border-color);
    }}

    [data-testid="stSidebar"] .block-container {{
        padding-top: 3rem;
    }}

    .sidebar-title {{
        font-size: 24px;
        font-weight: 800;
        background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 2rem;
        animation: slideInLeft 1s ease-out;
    }}

    /* Theme Toggle Button */
    .theme-toggle {{
        background: {glass_bg};
        backdrop-filter: blur(10px);
        border: 1px solid var(--border-color);
        border-radius: 50px;
        padding: 0.5rem 1rem;
        margin: 1rem auto 2rem;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: all 0.3s ease;
        width: fit-content;
    }}

    .theme-toggle:hover {{
        transform: scale(1.05);
        box-shadow: 0 5px 20px var(--hover-glow);
    }}

    /* =============== RADIO BUTTONS (Navigation) =============== */
    .stRadio > label {{
        font-weight: 600;
        color: var(--text-secondary);
        font-size: 15px;
    }}

    .stRadio > div {{
        gap: 0.5rem;
        background: {glass_bg};
        backdrop-filter: blur(8px);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid var(--border-color);
    }}

    .stRadio > div > label {{
        background: transparent;
        padding: 0.8rem 1rem;
        border-radius: 10px;
        transition: all 0.3s ease;
        cursor: pointer;
        border: 1px solid transparent;
    }}

    .stRadio > div > label:hover {{
        background: var(--bg-card);
        border-color: var(--accent-primary);
        transform: translateX(5px);
    }}

    .stRadio > div > label[data-baseweb="radio"] > div:first-child {{
        background-color: var(--accent-primary) !important;
    }}

    /* =============== BUTTONS =============== */
    .stButton > button {{
        background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 15px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    }}

    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.5);
        animation: glow 2s ease-in-out infinite;
    }}

    /* =============== SECTION TITLES =============== */
    .section-title {{
        font-size: clamp(20px, 3vw, 28px);
        font-weight: 800;
        color: var(--text-primary);
        margin: 2.5rem 0 1.5rem;
        position: relative;
        padding-left: 1rem;
        animation: slideInLeft 1s ease-out;
    }}

    .section-title::before {{
        content: '';
        position: absolute;
        left: 0;
        top: 50%;
        transform: translateY(-50%);
        width: 4px;
        height: 70%;
        background: linear-gradient(180deg, var(--accent-primary), var(--accent-secondary));
        border-radius: 2px;
    }}

    /* =============== DIVIDER =============== */
    hr {{
        margin: 2.5rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border-color), transparent);
    }}

    /* =============== SUCCESS/INFO BOXES =============== */
    .stSuccess, .stInfo, .stWarning {{
        background: {glass_bg} !important;
        backdrop-filter: blur(10px) !important;
        border-left: 4px solid var(--accent-primary) !important;
        border-radius: 12px !important;
        animation: slideInLeft 0.8s ease-out;
    }}

    /* =============== TOOLTIP =============== */
    .tooltip {{
        position: relative;
        display: inline-block;
        cursor: help;
    }}

    .tooltip .tooltiptext {{
        visibility: hidden;
        width: 200px;
        background-color: var(--bg-card);
        color: var(--text-primary);
        text-align: center;
        border-radius: 8px;
        padding: 0.75rem;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
        border: 1px solid var(--border-color);
        font-size: 13px;
    }}

    .tooltip:hover .tooltiptext {{
        visibility: visible;
        opacity: 1;
    }}

    /* =============== RESPONSIVE =============== */
    @media (max-width: 768px) {{
        .app-title {{
            font-size: 36px;
        }}
        
        .glass-card {{
            padding: 1.5rem;
        }}
        
        .feature-card {{
            padding: 1.5rem;
        }}
    }}

    /* =============== CUSTOM SCROLLBAR =============== */
    ::-webkit-scrollbar {{
        width: 10px;
        height: 10px;
    }}

    ::-webkit-scrollbar-track {{
        background: var(--bg-secondary);
    }}

    ::-webkit-scrollbar-thumb {{
        background: linear-gradient(180deg, var(--accent-primary), var(--accent-secondary));
        border-radius: 10px;
    }}

    ::-webkit-scrollbar-thumb:hover {{
        background: var(--accent-primary);
    }}
    </style>
    """

st.markdown(get_custom_css(st.session_state.dark_mode), unsafe_allow_html=True)

st.markdown('<div class="particles-bg"></div>', unsafe_allow_html=True)

def load_home_page():
    st.markdown("""
        <div class='app-header'>
            <h1 class='app-title'>Facial Emotion Recognition System</h1>
            <p class='app-subtitle'>
                A Deep Learning Application for Classifying Human Emotions Using Advanced CNN Models
            </p>
            <span class='badge'>✨ ResNet18 & MobileNetV2 Powered</span>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        img_path = "assets/fer_example.jpg"
        if os.path.exists(img_path):
            st.image(img_path,
                     caption="🎨 Real-time Emotion Detection",
                     use_container_width=True,
                     output_format="JPEG")
        else:
            st.markdown("""
                <div class='glass-card' style='text-align: center; padding: 4rem 2rem;'>
                    <div style='font-size: 64px; margin-bottom: 1rem;'>🖼️</div>
                    <p style='color: var(--text-secondary);'>
                        Place an example image at: <code>assets/fer_example.jpg</code>
                    </p>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("<h3 class='section-title'>✨ Key Features</h3>", unsafe_allow_html=True)
    
    features = [
        ("🎯", "Image Prediction", "Upload facial images and get instant emotion predictions with confidence scores"),
        ("📷", "Live Webcam", "Real-time emotion detection through your webcam with seamless processing"),
        ("🧠", "Model Inspector", "Deep dive into CNN architecture, layers, and parameter analysis"),
        ("📚", "Dataset Explorer", "Visualize training data, preprocessing steps, and data augmentation"),
        ("📈", "Training Logs", "Access comprehensive TensorBoard logs and training metrics"),
        ("🧪", "Model Evaluation", "Complete performance analysis on test dataset with confusion matrices")
    ]

    cols = st.columns(3)
    for idx, (icon, title, desc) in enumerate(features):
        with cols[idx % 3]:
            st.markdown(f"""
                <div class='feature-card'>
                    <div class='feature-icon'>{icon}</div>
                    <div class='feature-title'>{title}</div>
                    <div class='feature-desc'>{desc}</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("""
        <div class='glass-card'>
            <h3 style='margin-top: 0; color: var(--accent-primary);'>🔬 Technical Highlights</h3>
            <ul style='line-height: 2; color: var(--text-secondary);'>
                <li>🏗️ <strong>Architectures:</strong> ResNet18 & MobileNetV2 with transfer learning</li>
                <li>📊 <strong>Dataset:</strong> FER-2013 with 7 emotion classes (Happy, Sad, Angry, etc.)</li>
                <li>⚡ <strong>Performance:</strong> Real-time inference with GPU acceleration support</li>
                <li>🎨 <strong>Preprocessing:</strong> Face detection, alignment, and normalization pipeline</li>
                <li>📐 <strong>Input:</strong> 48x48 grayscale images | <strong>Output:</strong> 7-class softmax</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.success("👈 **Get Started:** Use the sidebar navigation to explore all features and start detecting emotions!")

    st.markdown("""
        <div style='text-align: center; margin-top: 3rem; padding: 2rem; color: var(--text-secondary); font-size: 14px;'>
            <p>Built with "THE BRO" team ❤️ using Streamlit & PyTorch</p>
            <p style='margin-top: 0.5rem; font-size: 12px;'>© 2025 Facial Emotion Recognition Project</p>
        </div>
    """, unsafe_allow_html=True)


def sidebar_navigation():
    with st.sidebar:
        st.markdown("<h2 class='sidebar-title'>🧭 Navigation</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🌙 Dark" if not st.session_state.dark_mode else "☀️ Light", use_container_width=True):
                st.session_state.dark_mode = not st.session_state.dark_mode
                st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        page = st.radio(
            "Select a Module:",
            [
                "🏠 Home",
                "📤 Upload Image Prediction",
                "📷 Webcam Live Detection",
                "🧠 Model Information",
                "📚 Dataset Explorer",
                "📊 Training & Evaluation Results"
            ],
            label_visibility="collapsed"
        )
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        st.markdown("""
            <div style='background: var(--glass-bg); backdrop-filter: blur(10px); 
                        border: 1px solid var(--border-color); border-radius: 12px; 
                        padding: 1rem; margin-top: 1rem;'>
                <p style='font-size: 12px; color: var(--text-secondary); margin: 0;'>
                    <strong>💡 Tip:</strong> Hover over feature cards for interactive effects!
                </p>
            </div>
        """, unsafe_allow_html=True)

    return page


def router(page):
    if page == "🏠 Home":
        load_home_page()

    elif page == "📤 Upload Image Prediction":
        from pages.image_prediction import image_prediction_page
        image_prediction_page()

    elif page == "📷 Webcam Live Detection":
        from pages.webcam_live import webcam_page
        webcam_page()

    elif page == "🧠 Model Information":
        from pages.Model_Info import load_model_info  # ✅ التصحيح هنا
        load_model_info()

    elif page == "📚 Dataset Explorer":
        from pages.dataset_page import dataset_page
        dataset_page()

    elif page == "📊 Training & Evaluation Results":
        from pages.evaluation_page import evaluation_page
        evaluation_page()


def main():
    page = sidebar_navigation()
    router(page)


if __name__ == "__main__":
    main()