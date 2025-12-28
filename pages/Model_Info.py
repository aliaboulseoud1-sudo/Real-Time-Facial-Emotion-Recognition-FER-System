import streamlit as st
import torch
import sys
import os
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from model import ModelBuilder
from config import Config

st.set_page_config(
    page_title="Model Info & Training",
    page_icon="📊",
    layout="wide"
)


@st.cache_data
def load_training_history(checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'history' in checkpoint:
            return checkpoint['history']
        return None
    except Exception as e:
        st.error(f"❌ Error loading checkpoint: {e}")
        return None

@st.cache_data
def load_tensorboard_logs(runs_dir):
    try:
        from tensorboard.backend.event_processing import event_accumulator
        
        run_dirs = [d for d in Path(runs_dir).iterdir() if d.is_dir()]
        if not run_dirs:
            return None
            
        latest_run = max(run_dirs, key=os.path.getmtime)
        
        ea = event_accumulator.EventAccumulator(str(latest_run))
        ea.Reload()
        
        metrics = {}
        for tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            metrics[tag] = {
                'steps': [e.step for e in events],
                'values': [e.value for e in events]
            }
        
        return metrics
    except ImportError:
        return None
    except Exception as e:
        st.warning(f"⚠️ Could not load TensorBoard logs: {e}")
        return None

@st.cache_resource
def load_model_info(model_name="resnet18", pretrained=True):
    try:
        model = ModelBuilder(
            num_classes=7,
            model_name=model_name,
            pretrained=pretrained,
            print_summary=False
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        layers_info = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  
                params = sum(p.numel() for p in module.parameters())
                trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
                if params > 0:  
                    layers_info.append({
                        'Layer': name if name else 'Root',
                        'Type': module.__class__.__name__,
                        'Parameters': params,
                        'Trainable': trainable == params
                    })
        
        return {
            'model': model,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'frozen_params': frozen_params,
            'layers': layers_info
        }
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

def plot_training_metrics(history):
    """Create interactive training metrics plots"""
    epochs = list(range(1, len(history['train_loss']) + 1))
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Loss (Train vs Val)', 'Accuracy (Train vs Val)',
                       'Learning Rate Schedule', 'Loss Comparison'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    fig.add_trace(
        go.Scatter(x=epochs, y=history['train_loss'], name='Train Loss',
                  line=dict(color='#FF6B6B', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=history['val_loss'], name='Val Loss',
                  line=dict(color='#4ECDC4', width=2)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=epochs, y=history['train_acc'], name='Train Acc',
                  line=dict(color='#95E1D3', width=2)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=history['val_acc'], name='Val Acc',
                  line=dict(color='#F38181', width=2)),
        row=1, col=2
    )
    
    if 'learning_rate' in history:
        fig.add_trace(
            go.Scatter(x=epochs, y=history['learning_rate'], name='LR',
                      line=dict(color='#AA96DA', width=2)),
            row=2, col=1
        )
    
    fig.add_trace(
        go.Scatter(x=epochs, y=history['train_loss'], name='Train',
                  line=dict(color='#FF6B6B', width=2), showlegend=False),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=history['val_loss'], name='Val',
                  line=dict(color='#4ECDC4', width=2), showlegend=False),
        row=2, col=2
    )
    
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    fig.update_xaxes(title_text="Epoch", row=2, col=2)
    
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy (%)", row=1, col=2)
    fig.update_yaxes(title_text="Learning Rate", row=2, col=1, type="log")
    fig.update_yaxes(title_text="Loss", row=2, col=2)
    
    fig.update_layout(
        height=800,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


st.title("📊 Model Info & Training Visualization")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🏋️ Training Metrics", "🧠 Model Architecture", "📈 TensorBoard Logs"])

with tab1:
    st.header("Training History")

    history_file = Path(r"results\training_history.json")

    @st.cache_data
    def load_training_history_from_file(file_path):
        try:
            import json
            with open(file_path, 'r') as f:
                history = json.load(f)
            return history
        except Exception as e:
            st.error(f"❌ Error loading training history from file: {e}")
            return None

    history = load_training_history_from_file(history_file)

    if history:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Best Train Loss",
                f"{min(history['train_loss']):.4f}",
                delta=f"Epoch {history['train_loss'].index(min(history['train_loss'])) + 1}"
            )
        with col2:
            st.metric(
                "Best Val Loss",
                f"{min(history['val_loss']):.4f}",
                delta=f"Epoch {history['val_loss'].index(min(history['val_loss'])) + 1}"
            )
        with col3:
            st.metric(
                "Best Train Acc",
                f"{max(history['train_acc']):.2f}%",
                delta=f"Epoch {history['train_acc'].index(max(history['train_acc'])) + 1}"
            )
        with col4:
            st.metric(
                "Best Val Acc",
                f"{max(history['val_acc']):.2f}%",
                delta=f"Epoch {history['val_acc'].index(max(history['val_acc'])) + 1}"
            )

        st.markdown("---")

        st.subheader("Training Curves")
        fig = plot_training_metrics(history)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Epoch-by-Epoch Metrics")
        metrics_df = pd.DataFrame({
            'Epoch': list(range(1, len(history['train_loss']) + 1)),
            'Train Loss': history['train_loss'],
            'Val Loss': history['val_loss'],
            'Train Acc (%)': history['train_acc'],
            'Val Acc (%)': history['val_acc']
        })
        if 'learning_rate' in history:
            metrics_df['Learning Rate'] = history['learning_rate']

        st.dataframe(
            metrics_df.style.highlight_min(subset=['Train Loss', 'Val Loss'], color='lightgreen')
                           .highlight_max(subset=['Train Acc (%)', 'Val Acc (%)'], color='lightblue'),
            use_container_width=True
        )

    else:
        st.warning("⚠️ No training history found in results folder")

with tab2:
    st.header("Model Architecture & Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox(
            "Model Type:",
            ["resnet18", "mobilenetv2"],
            index=0
        )
    
    with col2:
        pretrained = st.checkbox("Pretrained", value=True)
    
    with st.spinner(f"⏳ Loading {model_type}..."):
        model_info = load_model_info(model_type, pretrained)
    
    if model_info:
        st.success("✅ Model loaded successfully!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Parameters",
                f"{model_info['total_params']:,}",
                help="Total number of parameters in the model"
            )
        
        with col2:
            st.metric(
                "Trainable Parameters",
                f"{model_info['trainable_params']:,}",
                delta=f"{(model_info['trainable_params']/model_info['total_params']*100):.1f}%"
            )
        
        with col3:
            st.metric(
                "Frozen Parameters",
                f"{model_info['frozen_params']:,}",
                delta=f"{(model_info['frozen_params']/model_info['total_params']*100):.1f}%"
            )
        
        st.markdown("---")

        st.subheader("Parameter Distribution")
        
        fig = go.Figure(data=[go.Pie(
            labels=['Trainable', 'Frozen'],
            values=[model_info['trainable_params'], model_info['frozen_params']],
            hole=0.4,
            marker_colors=['#4ECDC4', '#FF6B6B']
        )])
        
        fig.update_layout(
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, width='stretch')
        
        st.subheader("Layer Details")
        
        layers_df = pd.DataFrame(model_info['layers'])
        
        st.dataframe(
            layers_df.style.apply(
                lambda x: ['background-color: lightgreen' if v else 'background-color: lightcoral' 
                           for v in x],
                subset=['Trainable']
            ),
            use_container_width=True,
            height=400
        )
        
        with st.expander("📋 Detailed Model Summary"):
            st.text(str(model_info['model']))

with tab3:
    st.header("TensorBoard Logs")
    
    config = Config()
    runs_dir = Path(r"results\logs")
    
    if runs_dir.exists():
        with st.spinner("⏳ Loading TensorBoard logs..."):
            metrics = load_tensorboard_logs(runs_dir)
        
        if metrics:
            st.success("✅ TensorBoard logs loaded successfully!")
            
            st.subheader("Available Metrics")
            metric_names = list(metrics.keys())
            st.write(f"Found **{len(metric_names)}** metric(s):")
            
            for metric_name in metric_names:
                with st.expander(f"📊 {metric_name}"):
                    data = metrics[metric_name]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=data['steps'],
                        y=data['values'],
                        mode='lines+markers',
                        name=metric_name,
                        line=dict(width=2)
                    ))
                    
                    fig.update_layout(
                        title=metric_name,
                        xaxis_title="Step",
                        yaxis_title="Value",
                        height=400,
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig, width='stretch') 
        else:
            st.info("📌 No TensorBoard logs found or tensorboard package not installed.")
            st.markdown("""
            **To enable TensorBoard logging:**
            ```bash
            pip install tensorboard
            ```
            Then run training with TensorBoard enabled in `train.py`
            """)
    else:
        st.warning(f"⚠️ Logs directory not found: {runs_dir}")
        st.info("💡 Create the directory or train the model to generate logs")

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    💡 Tip: Train your model first to see training metrics and logs
    </div>
    """,
    unsafe_allow_html=True
)