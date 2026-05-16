import argparse
import logging
import sys
import os
from pathlib import Path

# Add current directory to path to allow imports if run from root
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import numpy as np

from model import UNet
from dataset import LandsatDataset
from loss import PhysicsInformedLoss, CombinedLoss


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


def get_processed_dirs(cities, data_root='data', exclude_city=None):
    """Get list of processed directories for given cities."""
    dirs = []
    for city in cities:
        if exclude_city and city == exclude_city:
            continue
        d = os.path.join(data_root, city, 'processed')
        if os.path.exists(d):
            dirs.append(d)
    return dirs


def build_criterion(loss_type='physics', lambda_veg=0.1, lambda_temp=0.1):
    """Build loss function based on type."""
    if loss_type == 'bce':
        logging.info("Using BCEWithLogitsLoss")
        return nn.BCEWithLogitsLoss()
    elif loss_type == 'combined':
        logging.info("Using CombinedLoss (BCE + Dice)")
        return CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    elif loss_type == 'physics':
        logging.info(f"Using PhysicsInformedLoss (λ_veg={lambda_veg}, λ_temp={lambda_temp})")
        return PhysicsInformedLoss(
            bce_weight=0.5, dice_weight=0.5,
            lambda_veg=lambda_veg, lambda_temp=lambda_temp
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def train_net(net, device, data_dirs, epochs=20, batch_size=1, lr=1e-5, 
              save_cp=True, loss_type='physics', lambda_veg=0.1, lambda_temp=0.1,
              scheduler_type='cosine', iterations_per_epoch=20):
    
    # 1. Create dataset (enable physics data if using physics loss)
    use_physics = (loss_type == 'physics')
    dataset = LandsatDataset(data_dirs, return_physics=use_physics)
    
    n_train = len(dataset)
    if n_train == 0:
        logging.error("No training data found!")
        return
    
    train_set = dataset

    # 2. Create data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)

    # 3. Initialize logging
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Loss function:   {loss_type}
        Scheduler:       {scheduler_type}
        Device:          {device.type}
    ''')
    
    # 4. Optimizer & Loss
    optimizer = optim.Adam(net.parameters(), lr=lr)
    
    if scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    elif scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        scheduler = None
    
    criterion = build_criterion(loss_type, lambda_veg, lambda_temp)

    # 5. Training loop
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        
        with tqdm(total=n_train * iterations_per_epoch, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for _ in range(iterations_per_epoch):
                for batch in train_loader:
                    images = batch['image'].to(device=device, dtype=torch.float32)
                    true_masks = batch['mask'].to(device=device, dtype=torch.float32)

                    # Forward Pass
                    masks_pred = net(images)
                    
                    # Compute loss (with physics data if available)
                    if use_physics and 'ndvi' in batch and 'lst' in batch:
                        ndvi = batch['ndvi'].to(device=device, dtype=torch.float32)
                        lst = batch['lst'].to(device=device, dtype=torch.float32)
                        loss = criterion(masks_pred, true_masks, ndvi=ndvi, lst=lst)
                    elif hasattr(criterion, 'forward') and loss_type == 'combined':
                        loss = criterion(masks_pred, true_masks)
                    else:
                        loss = criterion(masks_pred, true_masks)

                    # Backward Pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    pbar.set_postfix(**{'loss': loss.item()})
                    epoch_loss += loss.item()
                    pbar.update(images.shape[0])
        
        # Step the scheduler
        if scheduler:
            scheduler.step()
        
        avg_loss = epoch_loss / (n_train * iterations_per_epoch)
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f'Epoch {epoch+1} finished ! Avg Loss: {avg_loss:.4f}, LR: {current_lr:.2e}')
        
        # Log to CSV
        log_file = 'training_log.csv'
        if not os.path.exists(log_file):
            with open(log_file, 'w') as f:
                f.write('epoch,loss,lr,loss_type\n')
        with open(log_file, 'a') as f:
            f.write(f'{epoch+1},{avg_loss},{current_lr},{loss_type}\n')
        
        if save_cp:
            Path('checkpoints/').mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), f'checkpoints/checkpoint_epoch{epoch + 1}.pth')
            # Save latest
            torch.save(net.state_dict(), 'checkpoints/latest_model.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train UHI detection model')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--loss', type=str, default='physics', 
                        choices=['bce', 'combined', 'physics'],
                        help='Loss function: bce, combined, or physics')
    parser.add_argument('--lambda-veg', type=float, default=0.1,
                        help='Weight for vegetation physics penalty (physics loss only)')
    parser.add_argument('--lambda-temp', type=float, default=0.1,
                        help='Weight for temperature physics penalty (physics loss only)')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'step', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--iterations', type=int, default=20,
                        help='Iterations per epoch (random crops per image)')
    parser.add_argument('--exclude-city', type=str, default=None,
                        help='City to exclude from training (for leave-one-out)')
    parser.add_argument('--cities', type=str, nargs='+', default=None,
                        help='Specific cities to train on (default: all discovered)')
    parser.add_argument('--data-root', type=str, default='data',
                        help='Root data directory')
    parser.add_argument('--checkpoint-name', type=str, default=None,
                        help='Custom name for saved checkpoint (e.g. "ablation_bce")')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Discover or use specified cities
    if args.cities:
        cities = args.cities
    else:
        cities = discover_cities(args.data_root)
    
    if not cities:
        logging.error(f"No cities with processed data found in {args.data_root}/")
        sys.exit(1)
    
    logging.info(f"Available cities: {cities}")
    
    processed_dirs = get_processed_dirs(cities, args.data_root, args.exclude_city)
    
    if args.exclude_city:
        logging.info(f"Excluding {args.exclude_city} (leave-one-out mode)")
    
    logging.info(f"Training on: {processed_dirs}")

    net = UNet(n_channels=3, n_classes=1, bilinear=True)
    net.to(device=device)

    try:
        train_net(
            net=net, device=device, data_dirs=processed_dirs, 
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            loss_type=args.loss, lambda_veg=args.lambda_veg, lambda_temp=args.lambda_temp,
            scheduler_type=args.scheduler, iterations_per_epoch=args.iterations
        )
        
        # Save with custom name if specified
        if args.checkpoint_name:
            Path('checkpoints/').mkdir(parents=True, exist_ok=True)
            custom_path = f'checkpoints/{args.checkpoint_name}.pth'
            torch.save(net.state_dict(), custom_path)
            logging.info(f'Saved custom checkpoint: {custom_path}')
            
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
