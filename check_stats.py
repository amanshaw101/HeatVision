import numpy as np
import os
import glob
from pathlib import Path
import pandas as pd

def compute_dataset_stats(data_root='data'):
    data_path = Path(data_root)
    cities = [d.name for d in sorted(data_path.iterdir()) if (d / 'processed').exists()]
    
    city_stats = []
    
    for city in cities:
        processed_dir = data_path / city / 'processed'
        lst_files = list(processed_dir.glob('*_LST.npy'))
        
        if not lst_files:
            continue
            
        base = lst_files[0].name.replace("_LST.npy", "")
        
        lst_data = np.load(processed_dir / f"{base}_LST.npy")
        ndvi_data = np.load(processed_dir / f"{base}_NDVI.npy")
        
        label_files = list(processed_dir.glob('*_WeakLabel.npy'))
        if label_files:
            uhi_data = np.load(label_files[0])
            uhi_frac = uhi_data.mean() * 100
        else:
            uhi_frac = 0.0
            
        valid = lst_data > 200.0
        valid_lst = lst_data[valid]
        valid_ndvi = ndvi_data[valid]
        
        h, w = lst_data.shape
        
        city_stats.append({
            "City": city,
            "Image Size": f"{h}×{w}",
            "Total Pixels": f"{h*w:,}",
            "Valid (%)": f"{valid.mean()*100:.1f}%",
            "LST Min (K)": f"{valid_lst.min():.1f}" if valid_lst.size > 0 else "N/A",
            "LST Max (K)": f"{valid_lst.max():.1f}" if valid_lst.size > 0 else "N/A",
            "LST Mean (K)": f"{valid_lst.mean():.1f}" if valid_lst.size > 0 else "N/A",
            "NDVI Mean": f"{valid_ndvi.mean():.3f}" if valid_ndvi.size > 0 else "N/A",
            "UHI (%)": f"{uhi_frac:.1f}%",
        })
        
    df = pd.DataFrame(city_stats)
    with open("stats_output.txt", "w", encoding="utf-8") as f:
        f.write(df.to_string(index=False))
    print("Stats written to stats_output.txt")

if __name__ == "__main__":
    compute_dataset_stats("C:/Users/robor/OneDrive/Desktop/dekhte hai/data")
