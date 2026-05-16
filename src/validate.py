"""
Validation Framework for UHI Detection.
Implements multiple validation strategies to break circular evaluation.

Strategies:
1. Spatial Hold-Out: Split image into train/test regions 
2. Alternative Weak Labels: Different thresholds to test generalization
3. LST Correlation: Physical consistency check (UHI ↔ high LST)
4. Leave-One-City-Out: Train on N-1 cities, test on held-out city

Usage:
    python src/validate.py --strategy all
    python src/validate.py --strategy leave-one-out --epochs 10
"""
import argparse
import json
import logging
import sys
import subprocess
from pathlib import Path
import numpy as np
import torch
import cv2
from scipy import stats

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from model import UNet
from preprocessing import compute_reflectance, generate_weak_labels
from evaluate import evaluate_city, compute_metrics, normalize


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


def load_city_data(city):
    """Load all processed data for a city."""
    processed_dir = Path(f"data/{city}/processed")
    lst_files = list(processed_dir.glob("*_LST.npy"))
    if not lst_files:
        return None
    
    base = lst_files[0].name.replace("_LST.npy", "")
    data = {
        'lst': np.load(processed_dir / f"{base}_LST.npy"),
        'b4': np.load(processed_dir / f"{base}_B4.npy"),
        'b5': np.load(processed_dir / f"{base}_B5.npy"),
        'ndvi': np.load(processed_dir / f"{base}_NDVI.npy"),
    }
    
    label_files = list(processed_dir.glob("*_WeakLabel.npy"))
    if label_files:
        data['weak_label'] = np.load(label_files[0])
    
    data['b4_ref'] = compute_reflectance(data['b4'], 2.0E-05, -0.100000, 45.0)
    return data


def validate_lst_correlation(cities, model_path='checkpoints/latest_model.pth'):
    """
    Strategy 3: LST Correlation.
    Check if model predictions correlate with LST (physical consistency).
    A physically meaningful model should predict higher UHI probability in hotter areas.
    """
    logging.info("\n=== LST Correlation Validation ===")
    
    device = torch.device('cpu')
    model = UNet(n_channels=3, n_classes=1, bilinear=True)
    if not Path(model_path).exists():
        logging.warning(f"No model found at {model_path}")
        return {}
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    results = {}
    for city in cities:
        data = load_city_data(city)
        if data is None:
            continue
        
        # Get model predictions
        prob_flat, gt_flat = evaluate_city(city, model, device, patch_size=256)
        if prob_flat is None:
            continue
        
        # For LST correlation, we need the full-resolution maps
        lst = data['lst']
        ndvi = data['ndvi']
        b4_ref = data['b4_ref']
        
        # Create evaluation mask (same as evaluate.py)
        valid_mask = lst > 200.0
        eval_mask = valid_mask & (ndvi <= 0.4) & (ndvi >= 0.0) & (b4_ref <= 0.25)
        
        lst_flat = lst[eval_mask].astype(np.float64)
        
        # Pearson correlation between predictions and LST
        if len(lst_flat) > 0 and len(prob_flat) == len(lst_flat):
            pearson_r, pearson_p = stats.pearsonr(prob_flat, lst_flat)
            spearman_r, spearman_p = stats.spearmanr(prob_flat, lst_flat)
            
            results[city] = {
                'pearson_r': float(pearson_r),
                'pearson_p': float(pearson_p),
                'spearman_r': float(spearman_r),
                'spearman_p': float(spearman_p),
                'interpretation': 'positive correlation = physically consistent' 
                                  if pearson_r > 0 else 'WARNING: negative correlation'
            }
            
            logging.info(f"  {city}: Pearson r={pearson_r:.4f} (p={pearson_p:.2e}), "
                        f"Spearman ρ={spearman_r:.4f} (p={spearman_p:.2e})")
    
    return results


def validate_alternative_labels(cities, model_path='checkpoints/latest_model.pth'):
    """
    Strategy 2: Alternative Weak Labels.
    Generate weak labels with different thresholds and see if model 
    trained on original labels generalizes to alternative labels.
    If it does, the model learned spatial patterns beyond just the threshold rules.
    """
    logging.info("\n=== Alternative Label Validation ===")
    
    device = torch.device('cpu')
    model = UNet(n_channels=3, n_classes=1, bilinear=True)
    if not Path(model_path).exists():
        logging.warning(f"No model found at {model_path}")
        return {}
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Alternative threshold configurations
    alt_configs = {
        'strict': {'alpha': 0.75, 'beta': 0.25, 'desc': 'Stricter (α=0.75, β=0.25)'},
        'relaxed': {'alpha': 0.25, 'beta': 0.35, 'desc': 'Relaxed (α=0.25, β=0.35)'},
    }
    
    results = {}
    
    for config_name, config in alt_configs.items():
        results[config_name] = {'description': config['desc'], 'cities': {}}
        
        for city in cities:
            data = load_city_data(city)
            if data is None:
                continue
            
            lst, ndvi, b4_ref = data['lst'], data['ndvi'], data['b4_ref']
            
            # Generate alternative weak labels
            valid_mask = lst > 200.0
            valid_lst = lst[valid_mask]
            if valid_lst.size == 0:
                continue
            
            temp_threshold = np.nanmean(valid_lst) + config['alpha'] * np.nanstd(valid_lst)
            alt_label = (
                valid_mask &
                (lst > temp_threshold) &
                (ndvi < config['beta']) &
                (ndvi > 0.0) &
                (b4_ref < 0.25)
            ).astype(np.float32)
            
            # Get model predictions
            prob_flat, _ = evaluate_city(city, model, device, patch_size=256)
            if prob_flat is None:
                continue
            
            # Evaluate against alternative labels
            eval_mask = valid_mask & (ndvi <= 0.4) & (ndvi >= 0.0) & (b4_ref <= 0.25)
            alt_flat = alt_label[eval_mask].astype(np.float64)
            
            if len(prob_flat) == len(alt_flat):
                metrics = compute_metrics(prob_flat, alt_flat)
                results[config_name]['cities'][city] = metrics
                
                logging.info(f"  {config_name}/{city}: AUPRC={metrics['auprc']:.4f}, "
                           f"ROC-AUC={metrics['roc_auc']:.4f}")
    
    return results


def validate_leave_one_out(cities, epochs=10):
    """
    Strategy 4: Leave-One-City-Out.
    Train on N-1 cities, evaluate on the held-out city.
    True test of cross-city generalization.
    """
    logging.info("\n=== Leave-One-City-Out Validation ===")
    
    results = {}
    
    for test_city in cities:
        train_cities = [c for c in cities if c != test_city]
        config_name = f'loo_{test_city.lower()}'
        
        logging.info(f"\n  Training without {test_city} (on {train_cities})...")
        
        # Train
        cmd = [
            sys.executable, 'src/train.py',
            '--epochs', str(epochs),
            '--loss', 'physics',
            '--exclude-city', test_city,
            '--checkpoint-name', config_name,
            '--scheduler', 'cosine',
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logging.error(f"Training failed for LOO-{test_city}")
            continue
        
        # Evaluate on held-out city
        device = torch.device('cpu')
        model = UNet(n_channels=3, n_classes=1, bilinear=True)
        ckpt_path = f'checkpoints/{config_name}.pth'
        
        if not Path(ckpt_path).exists():
            continue
        
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        
        prob_flat, gt_flat = evaluate_city(test_city, model, device, patch_size=256)
        if prob_flat is not None:
            metrics = compute_metrics(prob_flat, gt_flat)
            results[test_city] = {
                'train_cities': train_cities,
                'metrics': metrics,
            }
            logging.info(f"  LOO {test_city}: AUPRC={metrics['auprc']:.4f}, "
                        f"ROC-AUC={metrics['roc_auc']:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Validation framework')
    parser.add_argument('--strategy', type=str, default='all',
                        choices=['all', 'lst-correlation', 'alt-labels', 'leave-one-out'],
                        help='Validation strategy to run')
    parser.add_argument('--model', type=str, default='checkpoints/latest_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Epochs for leave-one-out training')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    cities = discover_cities()
    if not cities:
        logging.error("No processed city data found!")
        return
    
    logging.info(f"Cities: {cities}")
    
    validation_results = {}
    
    if args.strategy in ['all', 'lst-correlation']:
        validation_results['lst_correlation'] = validate_lst_correlation(cities, args.model)
    
    if args.strategy in ['all', 'alt-labels']:
        validation_results['alternative_labels'] = validate_alternative_labels(cities, args.model)
    
    if args.strategy in ['all', 'leave-one-out']:
        validation_results['leave_one_out'] = validate_leave_one_out(cities, args.epochs)
    
    # Save results
    out_path = Path('validation_results.json')
    with open(out_path, 'w') as f:
        json.dump(validation_results, f, indent=2)
    logging.info(f"\nValidation results saved to {out_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    if 'lst_correlation' in validation_results:
        print("\n--- LST Correlation (Physical Consistency) ---")
        for city, r in validation_results['lst_correlation'].items():
            print(f"  {city}: Pearson r = {r['pearson_r']:.4f} "
                  f"({'✅ consistent' if r['pearson_r'] > 0 else '⚠️ inconsistent'})")
    
    if 'alternative_labels' in validation_results:
        print("\n--- Alternative Label Generalization ---")
        for config, data in validation_results['alternative_labels'].items():
            print(f"  Config: {data['description']}")
            for city, m in data.get('cities', {}).items():
                print(f"    {city}: AUPRC={m['auprc']:.4f}, ROC-AUC={m['roc_auc']:.4f}")
    
    if 'leave_one_out' in validation_results:
        print("\n--- Leave-One-City-Out ---")
        for city, data in validation_results['leave_one_out'].items():
            m = data['metrics']
            print(f"  Test: {city} (trained on {data['train_cities']})")
            print(f"    AUPRC={m['auprc']:.4f}, ROC-AUC={m['roc_auc']:.4f}")
    
    print("="*70)


if __name__ == '__main__':
    main()
