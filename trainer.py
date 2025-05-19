import torch
import torch.nn as nn
from WaveVIT import SiameseReIDModel, WaveVIT
from Dataloader import Market1501Dataset, RandomIdentitySampler
from batch_triplet_loss import BatchAllTripletLoss
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm
import argparse
import os

def get_layerwise_lr_decay_params(model, base_lr=3e-5, decay=0.9):
    """Applies LLRD: assigns lower learning rates to lower layers of ViT"""
    layers = list(model.vit.blocks) + [model.vit.norm]
    param_groups = []

    for i, layer in enumerate(layers):
        lr = base_lr * (decay ** (len(layers) - i - 1))
        params = list(layer.parameters())
        param_groups.append({'params': params, 'lr': lr})
    
    head_params = []
    for name, param in model.named_parameters():
        if not name.startswith('vit'):
            head_params.append(param)

    param_groups.append({'params': head_params, 'lr': base_lr * 10}) 

    return param_groups

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
        transforms.RandomCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ])

    model = SiameseReIDModel(WaveVIT()).to(device)

    param_groups = get_layerwise_lr_decay_params(model.encoder, base_lr=3e-5, decay=0.9)
    optimizer = torch.optim.Adam(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    dataset = Market1501Dataset(args.dataset_dir, transform=train_transforms)
    sampler = RandomIdentitySampler(dataset.labels, num_instances=4, batch_size=32)
    loader = DataLoader(dataset, batch_size=8, sampler=sampler, num_workers=4)

    margin = 0.3
    best_loss = float('inf')
    patience = 3 
    wait = 0
    triplet_loss = BatchAllTripletLoss(margin)

    os.makedirs(args.model_save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        progress_bar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch}")

        for batch_idx, (images, labels) in progress_bar:
            try:
                images, labels = images.to(device), labels.to(device)
                embeddings = model(images)
                loss, pos_dist, neg_dist = triplet_loss(embeddings, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step(epoch + batch_idx / len(loader))
                total_loss += loss.item()

                progress_bar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    '+Dist': f"{pos_dist.item():.4f}",
                    '-Dist': f"{neg_dist.item():.4f}"
                })

            except Exception as e:
                print(f"\n Error at batch {batch_idx} in epoch {epoch}: {e}")
                break

        avg_loss = total_loss / len(loader)
        print(f"\n[Epoch {epoch}] Avg Loss: {avg_loss:.4f}\n")

        # Save model
        save_path = os.path.join(args.model_save_dir, f"model_epoch_{epoch}.pt")
        torch.save(model.state_dict(), save_path)
        print(f"✓ Model saved at {save_path}")

        # Early stopping logic
        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            wait = 0
            print(f"✓ Loss improved, best_loss updated to {best_loss:.4f}")
        else:
            wait += 1
            print(f"✗ No improvement in loss for {wait} epoch(s)")

        if wait >= patience:
            print(f"\n Early stopping triggered at epoch {epoch}. Best loss: {best_loss:.4f}")
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Siamese ReID model with WaveVIT")
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to Market1501 dataset')
    parser.add_argument('--model_save_dir', type=str, default='./saved_models', help='Directory to save models')

    args = parser.parse_args()
    train(args)
