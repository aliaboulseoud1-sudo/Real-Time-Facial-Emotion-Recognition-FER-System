import streamlit as st
import os
from PIL import Image
import pandas as pd
import json
from pathlib import Path
import base64
import io
import plotly.graph_objects as go
import plotly.express as px

def evaluation_page():
    
    BASE_RESULTS = Path("results")
    PLOTS = BASE_RESULTS / "plots"
    
    TEST_REPORT_CSV = BASE_RESULTS / "test_classification_report.csv"
    TEST_METRICS_JSON = BASE_RESULTS / "test_metrics.json"
    TRAINING_HISTORY_JSON = BASE_RESULTS / "training_history.json"
    
    CM_RAW = PLOTS / "test_confusion_matrix.png"
    CM_NORMALIZED = PLOTS / "test_confusion_matrix_normalized.png"
    ROC_IMAGE = PLOTS / "test_roc_curves.png"
    
    def check_results_exist():
        if not BASE_RESULTS.exists():
            return False, "Results folder not found"
        
        key_files = [TEST_METRICS_JSON, TEST_REPORT_CSV]
        if not any(f.exists() for f in key_files):
            return False, "No evaluation results found"
        
        return True, "Results loaded successfully"
    
    def read_json_safe(path: Path):
        try:
            if not path.exists():
                return None, f"File not found: {path.name}"
            
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data, None
        except json.JSONDecodeError as e:
            return None, f"Invalid JSON format: {str(e)}"
        except Exception as e:
            return None, f"Error reading file: {str(e)}"
    
    def read_csv_safe(path: Path):
        try:
            if not path.exists():
                return None, f"File not found: {path.name}"
            
            df = pd.read_csv(path)
            return df, None
        except Exception as e:
            return None, f"Error reading CSV: {str(e)}"
    
    def show_image_if_exists(path: Path, caption: str = None, use_column_width=True):
        if not path.exists():
            st.warning(f"⚠️ Image not found: `{path.name}`")
            return False
        
        try:
            img = Image.open(path)
            st.image(img, caption=caption, use_container_width=use_column_width)
            return True
        except Exception as e:
            st.error(f"❌ Failed to load image: {str(e)}")
            return False
    
    def create_download_link(file_path: Path, label: str, icon: str = "📥"):
        if not file_path.exists():
            return None
        
        try:
            with file_path.open("rb") as f:
                data = f.read()
            
            b64 = base64.b64encode(data).decode()
            
            if file_path.suffix == '.csv':
                mime = 'text/csv'
            elif file_path.suffix == '.json':
                mime = 'application/json'
            elif file_path.suffix == '.png':
                mime = 'image/png'
            else:
                mime = 'application/octet-stream'
            
            href = f'<a href="data:{mime};base64,{b64}" download="{file_path.name}" style="text-decoration:none;">{icon} {label}</a>'
            return href
        except Exception as e:
            return None
    
    def extract_metrics_comprehensive(metrics_json, report_csv):
        accuracy, precision, recall, f1 = None, None, None, None
        
        if metrics_json:
            try:
                if "overall" in metrics_json:
                    overall = metrics_json["overall"]
                    accuracy = overall.get("accuracy")
                    precision = overall.get("precision_macro") or overall.get("precision")
                    recall = overall.get("recall_macro") or overall.get("recall")
                    f1 = overall.get("f1_macro") or overall.get("f1") or overall.get("f1_score")
                
                if accuracy is None and "accuracy" in metrics_json:
                    accuracy = metrics_json.get("accuracy")
                
                if precision is None:
                    precision = (metrics_json.get("precision_macro") or 
                               metrics_json.get("precision") or
                               metrics_json.get("macro_precision"))
                
                if recall is None:
                    recall = (metrics_json.get("recall_macro") or 
                             metrics_json.get("recall") or
                             metrics_json.get("macro_recall"))
                
                if f1 is None:
                    f1 = (metrics_json.get("f1_macro") or 
                         metrics_json.get("f1") or 
                         metrics_json.get("f1_score") or
                         metrics_json.get("macro_f1"))
                
                for key in ["macro avg", "weighted avg", "macro_avg", "weighted_avg", "macro average"]:
                    if key in metrics_json:
                        avg_data = metrics_json[key]
                        if isinstance(avg_data, dict):
                            if precision is None:
                                precision = avg_data.get("precision")
                            if recall is None:
                                recall = avg_data.get("recall")
                            if f1 is None:
                                f1 = (avg_data.get("f1-score") or 
                                     avg_data.get("f1_score") or 
                                     avg_data.get("f1"))
                        break
                
                if (precision is None or recall is None or f1 is None):
                    emotion_keys = [k for k in metrics_json.keys() 
                                  if isinstance(metrics_json[k], dict) and 
                                  k not in ['overall', 'macro avg', 'weighted avg', 'accuracy']]
                    
                    if emotion_keys:
                        precisions, recalls, f1s = [], [], []
                        for emotion in emotion_keys:
                            data = metrics_json[emotion]
                            if isinstance(data, dict):
                                if 'precision' in data:
                                    precisions.append(data['precision'])
                                if 'recall' in data:
                                    recalls.append(data['recall'])
                                f1_val = data.get('f1-score') or data.get('f1_score') or data.get('f1')
                                if f1_val:
                                    f1s.append(f1_val)
                        
                        if precisions and precision is None:
                            precision = sum(precisions) / len(precisions)
                        if recalls and recall is None:
                            recall = sum(recalls) / len(recalls)
                        if f1s and f1 is None:
                            f1 = sum(f1s) / len(f1s)
                
            except Exception as e:
                st.warning(f"⚠️ Error extracting from JSON: {str(e)}")
        
        if report_csv is not None and not report_csv.empty:
            try:
                df = report_csv.copy()
                
                df.columns = [str(col).strip().lower() for col in df.columns]
                
                if len(df.columns) > 0:
                    class_col = df.columns[0]
                    
                    if accuracy is None:
                        accuracy_rows = df[df[class_col].astype(str).str.lower().str.contains('accuracy', na=False)]
                        if not accuracy_rows.empty:
                            for col in df.columns[1:]:
                                try:
                                    val = accuracy_rows.iloc[0][col]
                                    if pd.notna(val) and val != '':
                                        accuracy = float(val)
                                        break
                                except:
                                    continue
                    
                    for avg_type in ['macro avg', 'weighted avg', 'macro', 'weighted']:
                        avg_rows = df[df[class_col].astype(str).str.lower().str.contains(avg_type, na=False)]
                        
                        if not avg_rows.empty:
                            row_data = avg_rows.iloc[0]
                            
                            if len(df.columns) >= 4:
                                try:
                                    if precision is None and pd.notna(row_data.iloc[1]):
                                        precision = float(row_data.iloc[1])
                                    if recall is None and pd.notna(row_data.iloc[2]):
                                        recall = float(row_data.iloc[2])
                                    if f1 is None and pd.notna(row_data.iloc[3]):
                                        f1 = float(row_data.iloc[3])
                                except:
                                    pass
                            
                            if precision is not None and recall is not None and f1 is not None:
                                break
                
            except Exception as e:
                st.warning(f"⚠️ Error extracting from CSV: {str(e)}")
        
        return accuracy, precision, recall, f1
    
    def format_metric(val):
        if val is None:
            return "-"
        try:
            num = float(val)
            if num > 1:
                return f"{num:.2f}%"
            else:
                return f"{num:.4f}"
        except:
            return str(val) if val else "-"
    
    def get_metric_color(val):
        try:
            num = float(val)
            if num > 1:
                num = num / 100
            
            if num >= 0.9:
                return "#4CAF50"
            elif num >= 0.8:
                return "#8BC34A"
            elif num >= 0.7:
                return "#FFC107"
            elif num >= 0.6:
                return "#FF9800"
            else:
                return "#F44336"
        except:
            return "var(--text-secondary)"
    
    def get_metric_emoji(name):
        emojis = {
            "accuracy": "🎯",
            "precision": "🔍",
            "recall": "📊",
            "f1-score": "⚖️",
            "f1": "⚖️"
        }
        return emojis.get(name.lower(), "📈")
    
    st.markdown("""
        <div style='text-align: center;'>
            <h1 class='app-title'>📊 Performance Metrics</h1>
            <p class='app-subtitle'>Comprehensive analysis of model performance and evaluation results</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr style='margin: 2rem 0;'>", unsafe_allow_html=True)
    
    exists, msg = check_results_exist()
    
    if not exists:
        st.error(f"❌ {msg}")
        st.info("""
        💡 **How to generate results:**
        1. Ensure your model is trained
        2. Run the evaluation script: `python evaluate.py`
        3. Check that results are saved in the `results/` folder
        4. Refresh this page
        """)
        st.stop()
    
    with st.spinner("📂 Loading evaluation results..."):
        metrics_json, json_error = read_json_safe(TEST_METRICS_JSON)
        report_csv, csv_error = read_csv_safe(TEST_REPORT_CSV)
        history_json, history_error = read_json_safe(TRAINING_HISTORY_JSON)
    
    accuracy, precision, recall, f1 = extract_metrics_comprehensive(metrics_json, report_csv)
    
    accuracy_str = format_metric(accuracy)
    precision_str = format_metric(precision)
    recall_str = format_metric(recall)
    f1_str = format_metric(f1)
    
    has_metrics = any(m != "-" for m in [accuracy_str, precision_str, recall_str, f1_str])
    
    if not has_metrics:
        st.warning("⚠️ Could not extract metrics from available files")
        
        with st.expander("🔍 Debug Information - Click to see available data"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**JSON Content:**")
                if metrics_json:
                    st.json(metrics_json)
                else:
                    st.write("JSON file not found or empty")
            
            with col2:
                st.markdown("**CSV Content:**")
                if report_csv is not None:
                    st.dataframe(report_csv)
                else:
                    st.write("CSV file not found or empty")
    
    with st.sidebar:
        st.markdown("### 🎛️ Display Options")
        
        view_mode = st.radio(
            "Select view mode:",
            ["📊 Overview Dashboard", "🔍 Detailed Analysis", "📈 Training History"],
            index=0
        )
        
        st.markdown("---")
        
        st.markdown("### 💾 Quick Downloads")
        
        download_items = [
            (TEST_REPORT_CSV, "Classification Report (CSV)", "📄"),
            (TEST_METRICS_JSON, "Metrics (JSON)", "📊"),
            (CM_RAW, "Confusion Matrix (PNG)", "🖼️"),
            (ROC_IMAGE, "ROC Curves (PNG)", "📈")
        ]
        
        for file_path, label, icon in download_items:
            if file_path.exists():
                link = create_download_link(file_path, label, icon)
                if link:
                    st.markdown(link, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("""
            <div style='background: var(--glass-bg); padding: 1rem; border-radius: 8px; font-size: 13px;'>
                <strong>📌 Quick Tips:</strong><br>
                • Overview: Quick performance summary<br>
                • Detailed: Full classification report<br>
                • History: Training progress charts<br>
            </div>
        """, unsafe_allow_html=True)
    
    if view_mode == "📊 Overview Dashboard":
        st.markdown("<h2 class='section-title'>📈 Performance Metrics</h2>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        metrics_data = [
            ("Accuracy", accuracy_str, accuracy, "🎯"),
            ("Precision", precision_str, precision, "🔍"),
            ("Recall", recall_str, recall, "📊"),
            ("F1-Score", f1_str, f1, "⚖️")
        ]
        
        for col, (name, val_str, val_raw, icon) in zip([col1, col2, col3, col4], metrics_data):
            with col:
                color = get_metric_color(val_raw) if val_raw is not None else "var(--text-secondary)"
                
                st.markdown(f"""
                    <div class='glass-card' style='text-align: center; padding: 1.5rem;'>
                        <div style='font-size: 48px; margin-bottom: 0.5rem;'>{icon}</div>
                        <div style='font-size: 14px; font-weight: 600; color: var(--text-secondary); 
                                    text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.8rem;'>
                            {name}
                        </div>
                        <div style='font-size: 32px; font-weight: 700; color: {color}; line-height: 1;'>
                            {val_str}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<hr style='margin: 2rem 0;'>", unsafe_allow_html=True)
        
        st.markdown("<h2 class='section-title'>🔍 Confusion Matrices</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                <div class='glass-card'>
                    <h4 style='color: var(--accent-primary); margin-top: 0;'>📊 Raw Counts</h4>
                    <p style='color: var(--text-secondary); font-size: 14px;'>
                        Shows the actual number of predictions for each class
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            if show_image_if_exists(CM_RAW):
                link = create_download_link(CM_RAW, "Download Image", "📥")
                if link:
                    st.markdown(link, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class='glass-card'>
                    <h4 style='color: var(--accent-secondary); margin-top: 0;'>📈 Normalized</h4>
                    <p style='color: var(--text-secondary); font-size: 14px;'>
                        Shows the percentage distribution for better comparison
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            if show_image_if_exists(CM_NORMALIZED):
                link = create_download_link(CM_NORMALIZED, "Download Image", "📥")
                if link:
                    st.markdown(link, unsafe_allow_html=True)
        
        st.markdown("<hr style='margin: 2rem 0;'>", unsafe_allow_html=True)
        
        st.markdown("<h2 class='section-title'>📉 ROC Curves</h2>", unsafe_allow_html=True)
        
        st.markdown("""
            <div class='glass-card'>
                <p style='color: var(--text-secondary); margin: 0;'>
                    Receiver Operating Characteristic (ROC) curves show the trade-off between 
                    true positive rate and false positive rate for each emotion class.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        if show_image_if_exists(ROC_IMAGE, use_column_width=True):
            link = create_download_link(ROC_IMAGE, "Download ROC Curves", "📥")
            if link:
                st.markdown(link, unsafe_allow_html=True)
    
    elif view_mode == "🔍 Detailed Analysis":
        st.markdown("<h2 class='section-title'>📋 Classification Report</h2>", unsafe_allow_html=True)
        
        if report_csv is not None and not report_csv.empty:
            st.markdown("""
                <div class='glass-card'>
                    <p style='color: var(--text-secondary); margin: 0; font-size: 14px;'>
                        Detailed per-class performance metrics including precision, recall, and F1-score
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.dataframe(
                report_csv,
                use_container_width=True,
                hide_index=True,
                height=400
            )
            
            link = create_download_link(TEST_REPORT_CSV, "Download Full Report (CSV)", "📥")
            if link:
                st.markdown(link, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Classification report not available")
        
        st.markdown("<hr style='margin: 2rem 0;'>", unsafe_allow_html=True)
        
        st.markdown("<h2 class='section-title'>📊 Detailed Metrics (JSON)</h2>", unsafe_allow_html=True)
        
        if metrics_json:
            st.markdown("""
                <div class='glass-card'>
                    <p style='color: var(--text-secondary); margin: 0; font-size: 14px;'>
                        Raw metrics data in JSON format for programmatic access
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            with st.expander("🔍 View JSON Data", expanded=False):
                st.json(metrics_json)
            
            link = create_download_link(TEST_METRICS_JSON, "Download Metrics (JSON)", "📥")
            if link:
                st.markdown(link, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Metrics JSON not available")
        
        st.markdown("<hr style='margin: 2rem 0;'>", unsafe_allow_html=True)
        
        st.markdown("<h2 class='section-title'>📊 All Visualizations</h2>", unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["📊 Raw CM", "📈 Normalized CM", "📉 ROC Curves"])
        
        with tab1:
            show_image_if_exists(CM_RAW, "Raw Confusion Matrix")
        
        with tab2:
            show_image_if_exists(CM_NORMALIZED, "Normalized Confusion Matrix")
        
        with tab3:
            show_image_if_exists(ROC_IMAGE, "ROC Curves for All Classes")
    
    elif view_mode == "📈 Training History":
        st.markdown("<h2 class='section-title'>📈 Training Progress</h2>", unsafe_allow_html=True)
        
        if history_json and history_error is None:
            try:
                train_loss = history_json.get('train_loss', [])
                val_loss = history_json.get('val_loss', [])
                train_acc = history_json.get('train_acc', [])
                val_acc = history_json.get('val_acc', [])
                
                if not train_loss and not val_loss:
                    st.warning("⚠️ Training history is empty")
                else:
                    epochs = list(range(1, len(train_loss) + 1))
                    
                    st.markdown("### 📉 Loss Curves")
                    
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        fig_loss = go.Figure()
                        
                        if train_loss:
                            fig_loss.add_trace(go.Scatter(
                                x=epochs, y=train_loss,
                                mode='lines+markers',
                                name='Training Loss',
                                line=dict(color='#FF6B6B', width=3),
                                marker=dict(size=8)
                            ))
                        
                        if val_loss:
                            fig_loss.add_trace(go.Scatter(
                                x=epochs, y=val_loss,
                                mode='lines+markers',
                                name='Validation Loss',
                                line=dict(color='#4ECDC4', width=3),
                                marker=dict(size=8)
                            ))
                        
                        fig_loss.update_layout(
                            title="Training and Validation Loss",
                            xaxis_title="Epoch",
                            yaxis_title="Loss",
                            template="plotly_dark",
                            hovermode='x unified',
                            height=400
                        )
                        
                        st.plotly_chart(fig_loss, use_container_width=True)
                    
                    with col2:
                        st.markdown("""
                            <div class='glass-card' style='height: 100%;'>
                                <h4 style='color: var(--accent-primary); margin-top: 0;'>📊 Loss Stats</h4>
                        """, unsafe_allow_html=True)
                        
                        if train_loss:
                            st.metric("Final Train Loss", f"{train_loss[-1]:.4f}")
                            st.metric("Min Train Loss", f"{min(train_loss):.4f}")
                        
                        if val_loss:
                            st.metric("Final Val Loss", f"{val_loss[-1]:.4f}")
                            st.metric("Min Val Loss", f"{min(val_loss):.4f}")
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown("<hr style='margin: 2rem 0;'>", unsafe_allow_html=True)
                    
                    st.markdown("### 🎯 Accuracy Curves")
                    
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        fig_acc = go.Figure()
                        
                        if train_acc:
                            fig_acc.add_trace(go.Scatter(
                                x=epochs, y=train_acc,
                                mode='lines+markers',
                                name='Training Accuracy',
                                line=dict(color='#95E1D3', width=3),
                                marker=dict(size=8)
                            ))
                        
                        if val_acc:
                            fig_acc.add_trace(go.Scatter(
                                x=epochs, y=val_acc,
                                mode='lines+markers',
                                name='Validation Accuracy',
                                line=dict(color='#F38181', width=3),
                                marker=dict(size=8)
                            ))
                        
                        fig_acc.update_layout(
                            title="Training and Validation Accuracy",
                            xaxis_title="Epoch",
                            yaxis_title="Accuracy",
                            template="plotly_dark",
                            hovermode='x unified',
                            height=400
                        )
                        
                        st.plotly_chart(fig_acc, use_container_width=True)
                    
                    with col2:
                        st.markdown("""
                            <div class='glass-card' style='height: 100%;'>
                                <h4 style='color: var(--accent-secondary); margin-top: 0;'>🎯 Accuracy Stats</h4>
                        """, unsafe_allow_html=True)
                        
                        if train_acc:
                            st.metric("Final Train Acc", f"{train_acc[-1]:.4f}")
                            st.metric("Max Train Acc", f"{max(train_acc):.4f}")
                        
                        if val_acc:
                            st.metric("Final Val Acc", f"{val_acc[-1]:.4f}")
                            st.metric("Max Val Acc", f"{max(val_acc):.4f}")
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown("<hr style='margin: 2rem 0;'>", unsafe_allow_html=True)
                    
                    link = create_download_link(TRAINING_HISTORY_JSON, "Download Training History (JSON)", "📥")
                    if link:
                        st.markdown(link, unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"❌ Error displaying training history: {str(e)}")
                
                with st.expander("🔍 Debug: View raw history data"):
                    st.json(history_json)
        else:
            st.warning("⚠️ Training history not available")
            if history_error:
                st.error(f"Error: {history_error}")
            
            st.info("""
            💡 **Training history** is generated during model training and contains:
            - Loss curves (training & validation)
            - Accuracy curves (training & validation)
            - Per-epoch metrics
            
            Run `python train.py` to generate this data.
            """)
    
    st.markdown("<hr style='margin: 3rem 0 2rem 0;'>", unsafe_allow_html=True)
    
    with st.expander("ℹ️ About These Metrics"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Performance Metrics:**
            - **Accuracy**: Overall correctness of predictions
            - **Precision**: Accuracy of positive predictions
            - **Recall**: Ability to find all positive cases
            - **F1-Score**: Harmonic mean of precision and recall""")
    
    with col2:
        st.markdown("""
        **Visualizations:**
        - **Confusion Matrix**: Shows prediction vs actual labels
        - **ROC Curves**: Trade-off between TPR and FPR
        - **Training Curves**: Model learning progress over time
        """)

st.markdown("""
    <div style='text-align: center; margin-top: 2rem; padding: 2rem; 
                color: var(--text-secondary); font-size: 14px;
                border-top: 1px solid var(--border-color);'>
        <p style='margin: 0.5rem 0;'>📊 Comprehensive evaluation powered by scikit-learn & PyTorch</p>
        <p style='margin: 0.5rem 0; font-weight: 600; color: var(--accent-primary);'>
            © 2024 THE BRO Team - Facial Emotion Recognition System
        </p>
    </div>
""", unsafe_allow_html=True)