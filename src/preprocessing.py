import os
import rasterio
import numpy as np
import argparse
import glob
from pathlib import Path

def load_band(path):
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)

def get_mtl_value(mtl_path, parameter):
    with open(mtl_path, 'r') as f:
        for line in f:
            if parameter in line:
                return float(line.split('=')[1].strip())
    return None

def compute_reflectance(dn, mult, add, sun_elevation_deg):
    rho_prime = dn * mult + add
    # Correction for Sun Angle
    # rho = rho_prime / sin(sun_elevation)
    # sun_elevation is in degrees in MTL
    sin_theta = np.sin(np.radians(sun_elevation_deg))
    # Avoid division by zero
    sin_theta = np.maximum(sin_theta, 0.1)
    rho = rho_prime / sin_theta
    
    # FIX: Reflectance cannot be negative. Clamp to 0.
    # Also clamp to 1.0 just in case.
    return np.clip(rho, 0.0, 1.0)

def compute_radiance(dn, mult, add):
    return dn * mult + add

def compute_brightness_temp(radiance, k1, k2):
    # Avoid division by zero or log of zero/negative
    radiance = np.maximum(radiance, 0.1) 
    return k2 / np.log((k1 / radiance) + 1)

def compute_ndvi(red, nir):
    denom = nir + red
    # Avoid division by zero
    denom = np.maximum(denom, 0.0001)
    return (nir - red) / denom

def compute_emissivity(ndvi):
    # NDVI Threshold Method
    # NDVI < 0.2: Soil (0.97)
    # NDVI > 0.5: Vegetation (0.99)
    # 0.2 <= NDVI <= 0.5: Mixed
    
    epsilon = np.zeros_like(ndvi)
    epsilon[ndvi < 0.2] = 0.97
    epsilon[ndvi > 0.5] = 0.99
    
    mixed_mask = (ndvi >= 0.2) & (ndvi <= 0.5)
    pv = ((ndvi[mixed_mask] - 0.2) / (0.5 - 0.2)) ** 2
    epsilon[mixed_mask] = 0.99 * pv + 0.97 * (1 - pv) 
    
    return epsilon

def compute_lst(bt, emissivity, wavelength=10.8):
    # Planck's constant * speed of light / Boltzmann constant
    sigma = 14380.0
    return bt / (1 + (wavelength * bt / sigma) * np.log(emissivity))

def generate_weak_labels(lst, ndvi, b4_ref=None):
    # Physics-Guided Labeling (v2 - with nodata & desert handling)
    # UHI = High LST AND Low Vegetation AND Urban (not desert)
    
    # 0. Valid pixel mask: exclude nodata/fill (LST < 200K is physically impossible)
    valid_mask = lst > 200.0
    
    # 1. Temperature Threshold: computed ONLY on valid pixels
    valid_lst = lst[valid_mask]
    if valid_lst.size == 0:
        return np.zeros_like(lst, dtype=np.float32)
    temp_threshold = np.nanmean(valid_lst) + 0.5 * np.nanstd(valid_lst)
    
    # 2. Vegetation Threshold (Urban < 0.3 — tighter than before)
    veg_threshold = 0.3
    
    # 3. Desert/Sand Mask: bare sand has HIGH red reflectance (> 0.25)
    #    Urban impervious surfaces (concrete, asphalt) are darker in red band
    if b4_ref is not None:
        not_desert = b4_ref < 0.25
    else:
        not_desert = np.ones_like(lst, dtype=bool)
    
    # UHI Mask: Hot AND Not Vegetated AND Not Water AND Not Desert AND Valid
    uhi_mask = (
        valid_mask &
        (lst > temp_threshold) &
        (ndvi < veg_threshold) &
        (ndvi > 0.0) &
        not_desert
    )
    
    return uhi_mask.astype(np.float32)

def process_scene(scene_dir, output_dir):
    print(f"Processing scene in {scene_dir}...")
    
    # Find files
    mtl_files = glob.glob(os.path.join(scene_dir, "*_MTL.txt"))
    if not mtl_files:
        print(f"Error: No MTL file found in {scene_dir}")
        return
    mtl_path = mtl_files[0]
    
    b4_files = glob.glob(os.path.join(scene_dir, "*_B4.TIF"))
    b5_files = glob.glob(os.path.join(scene_dir, "*_B5.TIF"))
    b10_files = glob.glob(os.path.join(scene_dir, "*_B10.TIF"))
    
    if not (b4_files and b5_files and b10_files):
        print(f"Error: Missing band files in {scene_dir}")
        return
        
    # Load Metadata
    print("Loading Metadata...")
    # Thermal Constants
    rad_mult_b10 = get_mtl_value(mtl_path, "RADIANCE_MULT_BAND_10")
    rad_add_b10 = get_mtl_value(mtl_path, "RADIANCE_ADD_BAND_10")
    k1_b10 = get_mtl_value(mtl_path, "K1_CONSTANT_BAND_10")
    k2_b10 = get_mtl_value(mtl_path, "K2_CONSTANT_BAND_10")
    
    # Optical Constants (Reflectance)
    ref_mult_b4 = get_mtl_value(mtl_path, "REFLECTANCE_MULT_BAND_4")
    ref_add_b4 = get_mtl_value(mtl_path, "REFLECTANCE_ADD_BAND_4")
    ref_mult_b5 = get_mtl_value(mtl_path, "REFLECTANCE_MULT_BAND_5")
    ref_add_b5 = get_mtl_value(mtl_path, "REFLECTANCE_ADD_BAND_5")
    
    sun_elev = get_mtl_value(mtl_path, "SUN_ELEVATION")
    
    # Load Bands
    print("Loading Bands...")
    b4 = load_band(b4_files[0]) # Red (DN)
    b5 = load_band(b5_files[0]) # NIR (DN)
    b10 = load_band(b10_files[0]) # Thermal (DN)
    
    if b4.shape != b10.shape:
        print(f"Warning: resizing B10 {b10.shape} to match B4 {b4.shape}")
        import cv2
        b10 = cv2.resize(b10, (b4.shape[1], b4.shape[0]), interpolation=cv2.INTER_CUBIC)

    # 1. Radiance (Thermal)
    print("Computing Radiance...")
    L_lambda = compute_radiance(b10, rad_mult_b10, rad_add_b10)
    print(f"  Radiance Min: {L_lambda.min():.4f}, Max: {L_lambda.max():.4f}, Mean: {L_lambda.mean():.4f}")
    
    # 2. Brightness Temp
    print("Computing Brightness Temp...")
    bt = compute_brightness_temp(L_lambda, k1_b10, k2_b10)
    print(f"  BT (K) Min: {bt.min():.1f}, Max: {bt.max():.1f}, Mean: {bt.mean():.1f}")
    
    # 3. Reflectance (Optical) -> NDVI
    print("Computing TOA Reflectance & NDVI...")
    # Convert DN to TOA Reflectance
    print(f"  B4 (DN) Min: {b4.min()}, Max: {b4.max()}")
    print(f"  B5 (DN) Min: {b5.min()}, Max: {b5.max()}")
    
    b4_ref = compute_reflectance(b4, ref_mult_b4, ref_add_b4, sun_elev)
    b5_ref = compute_reflectance(b5, ref_mult_b5, ref_add_b5, sun_elev)
    print(f"  Red Refl Min: {b4_ref.min():.4f}, Max: {b4_ref.max():.4f}")
    print(f"  NIR Refl Min: {b5_ref.min():.4f}, Max: {b5_ref.max():.4f}")
    
    ndvi = compute_ndvi(b4_ref, b5_ref)
    print(f"  NDVI Min: {ndvi.min():.4f}, Max: {ndvi.max():.4f}, Mean: {ndvi.mean():.4f}")
    
    # 4. Emissivity
    print("Computing Emissivity...")
    emissivity = compute_emissivity(ndvi)
    
    # 5. LST
    print("Computing LST...")
    lst = compute_lst(bt, emissivity)
    print(f"  LST (K) Min: {lst.min():.1f}, Max: {lst.max():.1f}, Mean: {lst.mean():.1f}")
    
    # 6. Weak Labels (v2: with desert mask)
    print("Generating Weak Labels (v2)...")
    weak_labels = generate_weak_labels(lst, ndvi, b4_ref=b4_ref)
    print(f"  Weak Label fraction: {weak_labels.mean():.4f} ({weak_labels.sum():.0f} pixels)")
    
    # Save Outputs
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    scene_id = Path(scene_dir).name
    
    np.save(output_path / f"{scene_id}_LST.npy", lst)
    np.save(output_path / f"{scene_id}_NDVI.npy", ndvi)
    np.save(output_path / f"{scene_id}_WeakLabel.npy", weak_labels)
    # Save optical bands for training
    np.save(output_path / f"{scene_id}_B4.npy", b4)
    np.save(output_path / f"{scene_id}_B5.npy", b5)
    
    # Explicit Logging
    try:
        with open("preprocessing.log", "a") as f:
            f.write(f"--- Scene: {scene_id} ---\n")
            f.write(f"Radiance: Min={L_lambda.min():.4f}, Max={L_lambda.max():.4f}, Mean={L_lambda.mean():.4f}\n")
            f.write(f"BT (K): Min={bt.min():.2f}, Max={bt.max():.2f}, Mean={bt.mean():.2f}\n")
            # b4_ref/b5_ref might not be in scope if not defined? They are defined above.
            f.write(f"Refl B4: Min={b4_ref.min():.4f}, Max={b4_ref.max():.4f}\n")
            f.write(f"Refl B5: Min={b5_ref.min():.4f}, Max={b5_ref.max():.4f}\n")
            f.write(f"NDVI: Min={ndvi.min():.4f}, Max={ndvi.max():.4f}, Mean={ndvi.mean():.4f}\n")
            f.write(f"LST (K): Min={lst.min():.2f}, Max={lst.max():.2f}, Mean={lst.mean():.2f}\n")
    except Exception as e:
        print(f"Logging failed: {e}")

    print(f"Done! Saved preprocessed arrays to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", type=str, required=True, help="London, Dubai, or Mumbai")
    parser.add_argument("--data_dir", type=str, default="data", help="Root data directory")
    args = parser.parse_args()
    
    city_dir = os.path.join(args.data_dir, args.city)
    
    # Check if files are directly in city_dir or in subdirectories
    # We look for MTL files to identify scenes
    # Support LC08 (Landsat 8) and LC09 (Landsat 9)
    mtl_files = glob.glob(os.path.join(city_dir, "*_MTL.txt")) + \
                glob.glob(os.path.join(city_dir, "*", "*_MTL.txt"))
                
    if not mtl_files:
        print(f"No Landsat metadata (*_MTL.txt) found in {city_dir} or immediate subdirectories.")
        print("Structure should be: data/City/LC08.../ or data/City/ directly.")
    else:
        print(f"Found {len(mtl_files)} scenes for {args.city}.")
        for mtl_path in mtl_files:
            # The scene directory is the folder containing the MTL file
            scene_dir = os.path.dirname(mtl_path)
            process_scene(scene_dir, os.path.join(city_dir, "processed"))
