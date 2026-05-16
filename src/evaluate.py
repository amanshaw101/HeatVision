"""
Evaluate the trained UHI model on each city's data.
Computes: AUPRC, ROC-AUC, R², RMSE, IoU, F1 per city + cross-city validation.
Saves results to evaluation_metrics.json.
"""
import numpy as np
import torch
import cv2
import sys
import json
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    r2_score, mean_squared_error
)

sys.path.insert(0, str(Path(__file__).parent))
from model import UNet
from preprocessing import compute_reflectance

def normalize(arr):
    min_val = np.nanpercentile(arr, 2)
    max_val = np.nanpercentile(arr, 98)
    if max_val - min_val == 0:
        return np.zeros_like(arr)
    return np.clip((arr - min_val) / (max_val - min_val), 0, 1)

def evaluate_city(city, model, device, patch_size=256, apply_postprocessing=True):
    """Evaluate model on a city. Returns (prob_map_flat, gt_flat) for metrics."""
    processed_dir = Path(f"data/{city}/processed")
    lst_files = list(processed_dir.glob("*_LST.npy"))
    if not lst_files:
        return None, None
    
    base = lst_files[0].name.replace("_LST.npy", "")
    lst = np.load(processed_dir / f"{base}_LST.npy")
    b4 = np.load(processed_dir / f"{base}_B4.npy")
    b5 = np.load(processed_dir / f"{base}_B5.npy")
    ndvi = np.load(processed_dir / f"{base}_NDVI.npy")
    
    label_files = list(processed_dir.glob("*_WeakLabel.npy"))
    if not label_files:
        return None, None
    gt = np.load(label_files[0])
    
    b4_ref = compute_reflectance(b4, 2.0E-05, -0.100000, 45.0)
    valid_mask = lst > 200.0
    
    H, W = lst.shape
    prob_map = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)
    img_norm = np.stack([normalize(b4), normalize(b5), normalize(lst)], axis=0)
    
    stride = patch_size
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = img_norm[:, y:y+patch_size, x:x+patch_size]
            patch_tensor = torch.from_numpy(patch).unsqueeze(0).float().to(device)
            with torch.no_grad():
                output = model(patch_tensor)
                prob = torch.sigmoid(output).cpu().numpy()[0, 0]
            prob_map[y:y+patch_size, x:x+patch_size] += prob
            count_map[y:y+patch_size, x:x+patch_size] += 1.0
    
    count_map[count_map == 0] = 1
    prob_map /= count_map
    
    if apply_postprocessing:
        # Post-processing with physics constraints
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        prob_map = cv2.morphologyEx(prob_map, cv2.MORPH_OPEN, kernel)
        prob_map = cv2.GaussianBlur(prob_map, (5, 5), 0)
        prob_map[~valid_mask] = 0.0
        prob_map[ndvi > 0.4] = 0.0
        prob_map[ndvi < 0.0] = 0.0
        prob_map[b4_ref > 0.25] = 0.0
    
    # Only evaluate on valid pixels (exclude masked-out areas from both pred and GT)
    eval_mask = valid_mask & (ndvi <= 0.4) & (ndvi >= 0.0) & (b4_ref <= 0.25)
    
    prob_flat = prob_map[eval_mask].astype(np.float64)
    gt_flat = gt[eval_mask].astype(np.float64)
    
    return prob_flat, gt_flat

def compute_metrics(prob_flat, gt_flat):
    """Compute AUPRC, ROC-AUC, R², RMSE, IoU, F1 from predictions and labels."""
    gt_binary = (gt_flat > 0.5).astype(int)
    
    # Handle edge cases
    if len(np.unique(gt_binary)) < 2:
        return {
            "auprc": 0.0, "roc_auc": 0.0,
            "r2": 0.0, "rmse": 0.0,
            "iou": 0.0, "f1": 0.0,
            "best_threshold": 0.5,
            "n_pixels": int(len(gt_flat)),
            "pos_ratio": float(gt_binary.mean()),
        }
    
    # ROC-AUC
    roc_auc = roc_auc_score(gt_binary, prob_flat)
    
    # AUPRC (Average Precision)
    auprc = average_precision_score(gt_binary, prob_flat)
    
    # R² (coefficient of determination)
    r2 = r2_score(gt_flat, prob_flat)
    
    # RMSE
    rmse = float(np.sqrt(mean_squared_error(gt_flat, prob_flat)))
    
    # IoU and F1 at optimal threshold
    best_iou = 0.0
    best_f1 = 0.0
    best_threshold = 0.5
    
    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        pred_binary = (prob_flat > thresh).astype(int)
        
        tp = np.sum((pred_binary == 1) & (gt_binary == 1))
        fp = np.sum((pred_binary == 1) & (gt_binary == 0))
        fn = np.sum((pred_binary == 0) & (gt_binary == 1))
        
        # IoU = TP / (TP + FP + FN)
        iou = tp / (tp + fp + fn + 1e-8)
        
        # F1 = 2*TP / (2*TP + FP + FN)
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
        
        if f1 > best_f1:
            best_iou = iou
            best_f1 = f1
            best_threshold = thresh
    
    return {
        "auprc": float(auprc),
        "roc_auc": float(roc_auc),
        "r2": float(r2),
        "rmse": float(rmse),
        "iou": float(best_iou),
        "f1": float(best_f1),
        "best_threshold": float(best_threshold),
        "n_pixels": int(len(gt_flat)),
        "pos_ratio": float(gt_binary.mean() * 100),
    }


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


def main():
    print("Loading model...")
    device = torch.device('cpu')
    model = UNet(n_channels=3, n_classes=1, bilinear=True)
    
    ckpt_path = Path("checkpoints/latest_model.pth")
    if not ckpt_path.exists():
        print("No checkpoint found!")
        return
    
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    model.to(device)
    
    results = {}
    all_probs = []
    all_gts = []
    cross_city = {}
    
    # Discover cities dynamically
    city_list = discover_cities()
    if not city_list:
        print("No processed city data found!")
        return
    
    print(f"Evaluating cities: {city_list}")
    
    # Per-city evaluation
    for city in city_list:
        print(f"\nEvaluating {city}...")
        prob_flat, gt_flat = evaluate_city(city, model, device, patch_size=256)
        if prob_flat is not None:
            metrics = compute_metrics(prob_flat, gt_flat)
            results[city] = metrics
            all_probs.append(prob_flat)
            all_gts.append(gt_flat)
            
            print(f"  AUPRC:   {metrics['auprc']:.4f}")
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"  IoU:     {metrics['iou']:.4f}")
            print(f"  F1:      {metrics['f1']:.4f}")
            print(f"  R²:      {metrics['r2']:.4f}")
            print(f"  RMSE:    {metrics['rmse']:.4f}")
    
    # Cross-city validation
    print("\n--- Cross-City Validation ---")
    print(f"(Model trained on all {len(city_list)} cities, per-city breakdown)")
    
    for test_city in city_list:
        train_cities = [c for c in city_list if c != test_city]
        
        train_probs = []
        train_gts = []
        for tc in train_cities:
            if tc in results:
                idx = city_list.index(tc)
                if idx < len(all_probs):
                    train_probs.append(all_probs[idx])
                    train_gts.append(all_gts[idx])
        
        if train_probs:
            train_prob_all = np.concatenate(train_probs)
            train_gt_all = np.concatenate(train_gts)
            train_metrics = compute_metrics(train_prob_all, train_gt_all)
            
            test_metrics = results.get(test_city, {})
            
            cross_city[test_city] = {
                "test_auprc": test_metrics.get("auprc", 0),
                "test_roc_auc": test_metrics.get("roc_auc", 0),
                "test_iou": test_metrics.get("iou", 0),
                "train_auprc": train_metrics["auprc"],
                "train_roc_auc": train_metrics["roc_auc"],
                "generalization_gap_auprc": train_metrics["auprc"] - test_metrics.get("auprc", 0),
                "generalization_gap_roc": train_metrics["roc_auc"] - test_metrics.get("roc_auc", 0),
            }
            print(f"  Test: {test_city} | Train: {train_cities}")
            print(f"    Test AUPRC: {test_metrics.get('auprc', 0):.4f} | "
                  f"Train AUPRC: {train_metrics['auprc']:.4f} | "
                  f"Gap: {cross_city[test_city]['generalization_gap_auprc']:.4f}")
    
    results["cross_city"] = cross_city
    
    # Save results
    out_path = Path("evaluation_metrics.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
