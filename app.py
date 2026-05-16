import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from pathlib import Path
import tempfile
import os
import cv2
import json
import pandas as pd

# Import project modules
import sys
sys.path.append('src')
from model import UNet
from preprocessing import compute_radiance, compute_brightness_temp, compute_ndvi, compute_emissivity, compute_lst, generate_weak_labels, compute_reflectance

# Constants (Landsat 8 Band 10) for demo purposes if MTL is missing
DEFAULT_K1 = 774.8853
DEFAULT_K2 = 1321.0789
DEFAULT_RAD_MULT = 3.3420E-04
DEFAULT_RAD_ADD = 0.10000


def discover_cities(data_root='data'):
    """Auto-discover all cities with processed data."""
    cities = []
    data_path = Path(data_root)
    if data_path.exists():
        for city_dir in sorted(data_path.iterdir()):
            processed = city_dir / 'processed'
            if processed.exists() and list(processed.glob('*_B4.npy')):
                cities.append(city_dir.name)
    return cities


@st.cache_resource
def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=3, n_classes=1, bilinear=True)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        return None
    model.to(device)
    model.eval()
    return model

def normalize(arr):
    min_val = np.nanpercentile(arr, 2)
    max_val = np.nanpercentile(arr, 98)
    
    if max_val - min_val == 0:
        return np.zeros_like(arr)
        
    normalized = (arr - min_val) / (max_val - min_val)
    return np.clip(normalized, 0, 1)

def process_and_predict(b4_file, b5_file, b10_file, mtl_file, model):
    # 1. Read Files
    with rasterio.open(b4_file) as src:
        b4 = src.read(1).astype(np.float32)
    with rasterio.open(b5_file) as src:
        b5 = src.read(1).astype(np.float32)
    with rasterio.open(b10_file) as src:
        b10 = src.read(1).astype(np.float32)
        
    # Resize B10 if needed
    if b10.shape != b4.shape:
        b10 = cv2.resize(b10, (b4.shape[1], b4.shape[0]), interpolation=cv2.INTER_CUBIC)

    # 2. Parse MTL
    k1, k2 = DEFAULT_K1, DEFAULT_K2
    rad_mult, rad_add = DEFAULT_RAD_MULT, DEFAULT_RAD_ADD
    ref_mult_b4, ref_add_b4 = 2.0E-05, -0.100000
    ref_mult_b5, ref_add_b5 = 2.0E-05, -0.100000
    sun_elev = 45.0
    
    if mtl_file:
        string_data = mtl_file.getvalue().decode("utf-8")
        for line in string_data.split('\n'):
            if "RADIANCE_MULT_BAND_10" in line: rad_mult = float(line.split('=')[1].strip())
            if "RADIANCE_ADD_BAND_10" in line: rad_add = float(line.split('=')[1].strip())
            if "K1_CONSTANT_BAND_10" in line: k1 = float(line.split('=')[1].strip())
            if "K2_CONSTANT_BAND_10" in line: k2 = float(line.split('=')[1].strip())
            if "REFLECTANCE_MULT_BAND_4" in line: ref_mult_b4 = float(line.split('=')[1].strip())
            if "REFLECTANCE_ADD_BAND_4" in line: ref_add_b4 = float(line.split('=')[1].strip())
            if "REFLECTANCE_MULT_BAND_5" in line: ref_mult_b5 = float(line.split('=')[1].strip())
            if "REFLECTANCE_ADD_BAND_5" in line: ref_add_b5 = float(line.split('=')[1].strip())
            if "SUN_ELEVATION" in line: sun_elev = float(line.split('=')[1].strip())

    # 3. Physics Processing
    L_lambda = compute_radiance(b10, rad_mult, rad_add)
    bt = compute_brightness_temp(L_lambda, k1, k2)
    b4_ref = compute_reflectance(b4, ref_mult_b4, ref_add_b4, sun_elev)
    b5_ref = compute_reflectance(b5, ref_mult_b5, ref_add_b5, sun_elev)
    
    ndvi = compute_ndvi(b4_ref, b5_ref)
    epsilon = compute_emissivity(ndvi)
    lst = compute_lst(bt, epsilon)
    
    # 4. Prepare for Model
    img_tensor = np.stack([normalize(b4), normalize(b5), normalize(lst)], axis=0)
    img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).float()
    
    H, W = img_tensor.shape[2], img_tensor.shape[3]
    if H > 2048 or W > 2048:
        st.warning(f"Image is large ({H}x{W}). Cropping center 1024x1024 for quick demo.")
        cy, cx = H // 2, W // 2
        img_tensor = img_tensor[:, :, cy-512:cy+512, cx-512:cx+512]
        lst = lst[cy-512:cy+512, cx-512:cx+512]
        ndvi = ndvi[cy-512:cy+512, cx-512:cx+512]
        b4 = b4[cy-512:cy+512, cx-512:cx+512]
        b4_ref = b4_ref[cy-512:cy+512, cx-512:cx+512]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    img_tensor = img_tensor.to(device)
    
    debug_stats = {
        "L_lambda": {"min": L_lambda.min(), "max": L_lambda.max(), "mean": L_lambda.mean()},
        "bt": {"min": bt.min(), "max": bt.max(), "mean": bt.mean()},
        "b4_ref": {"min": b4_ref.min(), "max": b4_ref.max()},
        "b5_ref": {"min": b5_ref.min(), "max": b5_ref.max()},
        "ndvi": {"min": ndvi.min(), "max": ndvi.max(), "mean": ndvi.mean()},
        "lst": {"min": lst.min(), "max": lst.max(), "mean": lst.mean()}
    }

    with torch.no_grad():
        output = model(img_tensor)
        prob_map = torch.sigmoid(output).cpu().numpy()[0, 0]
        
    valid_mask = lst > 200.0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    prob_map = cv2.morphologyEx(prob_map, cv2.MORPH_OPEN, kernel)
    prob_map = cv2.GaussianBlur(prob_map, (5, 5), 0)
    prob_map[~valid_mask] = 0.0
    prob_map[ndvi > 0.4] = 0.0
    prob_map[ndvi < 0.0] = 0.0
    prob_map[b4_ref > 0.25] = 0.0
    
    return b4, ndvi, lst, prob_map, debug_stats


# --- Streamlit UI ---
st.set_page_config(page_title="Urban Heat Island Detection Using U-Net", layout="wide")

st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    h1 {
        font-family: 'Times New Roman', serif;
        color: #2c3e50;
    }
    h2, h3 {
        font-family: 'Arial', sans-serif;
        color: #34495e;
    }
    .metric-box {
        background-color: white;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        text-align: center;
    }
    .abstract {
        background-color: white;
        padding: 20px;
        border-left: 5px solid #3498db;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛰️ Urban Heat Island Detection Using U-Net")

# Discover cities dynamically
all_cities = discover_cities()
if not all_cities:
    all_cities = ["London", "Dubai", "Mumbai"]

# Tabs
tab1, tab2 = st.tabs([
    "🌍 Interactive Analysis", 
    "📊 Research Metrics"
])

# ===================== TAB 2: Research Metrics =====================
with tab2:
    st.header("Dataset Statistics & Model Metrics")
    
    # --- Per-City Dataset Stats ---
    st.subheader("📁 Per-City Dataset Overview")
    
    city_stats = []
    for c in all_cities:
        processed_dir = Path(f"data/{c}/processed")
        lst_files = list(processed_dir.glob("*_LST.npy"))
        if not lst_files:
            continue
        base = lst_files[0].name.replace("_LST.npy", "")
        lst_data = np.load(processed_dir / f"{base}_LST.npy")
        ndvi_data = np.load(processed_dir / f"{base}_NDVI.npy")
        
        valid = lst_data > 200.0
        valid_lst = lst_data[valid]
        valid_ndvi = ndvi_data[valid]
        
        label_files = list(processed_dir.glob("*_WeakLabel.npy"))
        uhi_frac = np.load(label_files[0]).mean() * 100 if label_files else 0.0
        
        city_stats.append({
            "City": c,
            "Image Size": f"{lst_data.shape[0]}×{lst_data.shape[1]}",
            "Valid Pixels (%)": f"{valid.mean()*100:.1f}%",
            "LST Min (K)": f"{valid_lst.min():.1f}" if valid_lst.size > 0 else "N/A",
            "LST Max (K)": f"{valid_lst.max():.1f}" if valid_lst.size > 0 else "N/A",
            "LST Mean (K)": f"{valid_lst.mean():.1f}" if valid_lst.size > 0 else "N/A",
            "NDVI Mean": f"{valid_ndvi.mean():.3f}" if valid_ndvi.size > 0 else "N/A",
        })
    
    if city_stats:
        df_stats = pd.DataFrame(city_stats)
        st.dataframe(df_stats, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # --- Comparison Charts ---
    st.subheader("📊 Cross-City Comparison")
    
    if city_stats:
        chart_data = []
        for c in all_cities:
            processed_dir = Path(f"data/{c}/processed")
            lst_files = list(processed_dir.glob("*_LST.npy"))
            if not lst_files:
                continue
            base = lst_files[0].name.replace("_LST.npy", "")
            lst_data = np.load(processed_dir / f"{base}_LST.npy")
            ndvi_data = np.load(processed_dir / f"{base}_NDVI.npy")
            valid = lst_data > 200.0
            valid_lst = lst_data[valid]
            valid_ndvi = ndvi_data[valid]
            
            label_files = list(processed_dir.glob("*_WeakLabel.npy"))
            uhi_frac = np.load(label_files[0]).mean() * 100 if label_files else 0.0
            
            chart_data.append({
                "City": c,
                "Mean LST (°C)": float(valid_lst.mean() - 273.15) if valid_lst.size > 0 else 0,
                "Mean NDVI": float(valid_ndvi.mean()) if valid_ndvi.size > 0 else 0,
                "Vegetation (NDVI>0.3) %": float((valid_ndvi > 0.3).mean() * 100) if valid_ndvi.size > 0 else 0,
                "Water (NDVI<0) %": float((valid_ndvi < 0).mean() * 100) if valid_ndvi.size > 0 else 0,
            })
        
        df_chart = pd.DataFrame(chart_data)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🌡️ Mean Land Surface Temperature**")
            st.bar_chart(df_chart.set_index("City")["Mean LST (°C)"])
        with col2:
            st.markdown("**🌿 Mean NDVI (Vegetation Index)**")
            st.bar_chart(df_chart.set_index("City")["Mean NDVI"])
        
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("**🏞️ Land Cover Distribution**")
            land_cover = df_chart[["City", "Vegetation (NDVI>0.3) %", "Water (NDVI<0) %"]].set_index("City")
            st.bar_chart(land_cover)
    
    st.markdown("---")
    
    # --- Model Performance Metrics ---
    st.subheader("🎯 Model Performance Metrics")
    
    metrics_file = Path("evaluation_metrics.json")
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            eval_metrics = json.load(f)
        
        cross_city_data = eval_metrics.pop("cross_city", {})
        
        metrics_rows = []
        chart_rows = []
        for city_name, m in eval_metrics.items():
            if 'auprc' not in m:
                continue
            metrics_rows.append({
                "City": city_name,
                "IoU": f"{m.get('iou', 0):.4f}",
                "F1-Score": f"{m.get('f1', 0):.4f}",
                "AUPRC": f"{m['auprc']:.4f}",
                "ROC-AUC": f"{m['roc_auc']:.4f}",
            })
            chart_rows.append({
                "City": city_name,
                "IoU": m.get("iou", 0),
                "F1-Score": m.get("f1", 0),
                "AUPRC": m["auprc"],
                "ROC-AUC": m["roc_auc"],
            })
        
        df_metrics = pd.DataFrame(metrics_rows)
        st.dataframe(df_metrics, use_container_width=True, hide_index=True)
        
        df_chart = pd.DataFrame(chart_rows)
        
        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown("**📈 AUPRC**")
            st.bar_chart(df_chart.set_index("City")["AUPRC"])
        with mc2:
            st.markdown("**📉 ROC-AUC**")
            st.bar_chart(df_chart.set_index("City")["ROC-AUC"])
        
        mc3, mc4 = st.columns(2)
        with mc3:
            st.markdown("**🎯 IoU (Intersection over Union)**")
            st.bar_chart(df_chart.set_index("City")["IoU"])
        with mc4:
            st.markdown("**📊 F1 Score**")
            st.bar_chart(df_chart.set_index("City")["F1-Score"])
        
        st.caption("Metrics computed against physics-guided weak labels using sliding-window inference (256×256 patches).")
    else:
        st.info("No evaluation metrics found. Run `python src/evaluate.py` to generate metrics.")
    
    st.markdown("---")
    
    # Model Info
    st.subheader("🧠 Model Training Summary")
    
    model_col1, model_col2 = st.columns(2)
    with model_col1:
        st.markdown("**Architecture**")
        st.markdown("""
        | Parameter | Value |
        |-----------|-------|
        | Model | U-Net |
        | Input Channels | 3 (B4, B5, LST) |
        | Output | 1 (UHI Probability) |
        | Activation | Sigmoid |
        """)
    
    with model_col2:
        st.markdown("**Training Configuration**")
        ckpt_dir = Path("checkpoints")
        ckpts = sorted(ckpt_dir.glob("checkpoint_epoch*.pth")) if ckpt_dir.exists() else []
        latest = ckpt_dir / "latest_model.pth" if ckpt_dir.exists() else None
        
        train_info = {
            "Learning Rate": "1e-5",
            "Batch Size": "1 (patch-based)",
            "Loss Function": "PhysicsInformedLoss (BCE+Dice+Physics)",
            "Optimizer": "Adam",
            "Scheduler": "CosineAnnealingLR",
            "Epochs Trained": str(len(ckpts)) if ckpts else "N/A",
            "Checkpoint Available": "✅" if latest and latest.exists() else "❌",
        }
        info_df = pd.DataFrame(list(train_info.items()), columns=["Parameter", "Value"])
        st.dataframe(info_df, use_container_width=True, hide_index=True)
    
    log_file = 'training_log.csv'
    if os.path.exists(log_file):
        try:
            df_log = pd.read_csv(log_file)
            st.markdown("**📉 Training Loss Curve**")
            st.line_chart(df_log.set_index('epoch')['loss'])
        except Exception:
            pass





# ===================== TAB 1: Interactive Analysis =====================
with tab1:
    st.header("Interactive Analysis")
    city = st.selectbox("Select City", all_cities)
    
    model_path = "checkpoints/latest_model.pth" 
    model = load_model(model_path)
    if model is None:
        st.error(f"Model not found at {model_path}. Please train the model first.")
    
    processed_dir = Path(f"data/{city}/processed")
    if not processed_dir.exists():
        st.warning(f"No processed data found for {city}.")
    else:
        lst_files = list(processed_dir.glob("*_LST.npy"))
        if lst_files:
            sample_base = lst_files[0].name.replace("_LST.npy", "")
            
            lst = np.load(processed_dir / f"{sample_base}_LST.npy")
            b4 = np.load(processed_dir / f"{sample_base}_B4.npy")
            b5 = np.load(processed_dir / f"{sample_base}_B5.npy")
            ndvi = np.load(processed_dir / f"{sample_base}_NDVI.npy")
            
            # Smart Crop
            H, W = lst.shape
            crop_h, crop_w = 1200, 1200
            if H > crop_h or W > crop_w:
                b4_ref_full = compute_reflectance(b4, 2.0E-05, -0.100000, 45.0)
                urban_mask_full = (b4_ref_full < 0.25) & (lst > 200.0)
                
                if urban_mask_full.any():
                    rows, cols = np.where(urban_mask_full)
                    cy = int(np.median(rows))
                    cx = int(np.median(cols))
                else:
                    cy, cx = H // 2, W // 2
                
                cy = max(crop_h // 2, min(cy, H - crop_h // 2))
                cx = max(crop_w // 2, min(cx, W - crop_w // 2))
                
                lst = lst[cy-600:cy+600, cx-600:cx+600]
                b4 = b4[cy-600:cy+600, cx-600:cx+600]
                b5 = b5[cy-600:cy+600, cx-600:cx+600]
                ndvi = ndvi[cy-600:cy+600, cx-600:cx+600]

            b4_ref_approx = compute_reflectance(b4, 2.0E-05, -0.100000, 45.0)
            valid_mask = lst > 200.0

            img_tensor = np.stack([normalize(b4), normalize(b5), normalize(lst)], axis=0)
            img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).float()
            
            if model:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model.to(device)
                img_tensor = img_tensor.to(device)
                with torch.no_grad():
                    output = model(img_tensor)
                    prob = torch.sigmoid(output).cpu().numpy()[0, 0]
                    
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                prob = cv2.morphologyEx(prob, cv2.MORPH_OPEN, kernel)
                prob = cv2.GaussianBlur(prob, (5, 5), 0)
                prob[~valid_mask] = 0.0
                prob[ndvi > 0.4] = 0.0
                prob[ndvi < 0.0] = 0.0
                prob[b4_ref_approx > 0.25] = 0.0
            else:
                prob = np.zeros_like(lst)

            def plot_with_colorbar(arr, title, cmap='viridis', vmin=None, vmax=None, figsize=(6, 5)):
                fig, ax = plt.subplots(figsize=figsize)
                if np.nanmin(arr) == 0 and np.nanmax(arr) > 0:
                     masked_arr = np.ma.masked_where(arr == 0, arr)
                     im = ax.imshow(masked_arr, cmap=cmap, vmin=vmin, vmax=vmax)
                else:
                     im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                if 'NDVI' in title: cbar.set_label('Index Value')
                elif 'LST' in title: cbar.set_label('Temperature (K)')
                elif 'Prob' in title: cbar.set_label('Probability')
                ax.set_title(title)
                ax.axis('off')
                return fig

            st.markdown("### 🚨 Urban Heat Island (UHI) Risk Map")
            main_col1, main_col2, main_col3 = st.columns([1, 4, 1])
            with main_col2:
                st.pyplot(plot_with_colorbar(prob, "UHI Risk (Probability)", cmap='jet', vmin=0, vmax=1, figsize=(8, 6)))

            st.markdown("---")
            st.markdown("### 📡 Input Satellite Data")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.pyplot(plot_with_colorbar(normalize(b4), "Optical (Red Band)", cmap='gray'))
            with col2:
                st.pyplot(plot_with_colorbar(lst, "Thermal (LST)", cmap='inferno'))
            with col3:
                st.pyplot(plot_with_colorbar(ndvi, "Vegetation (NDVI)", cmap='RdYlGn', vmin=-0.2, vmax=0.6))


