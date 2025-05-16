import torch
import torch.nn as nn
import timm
from kymatio import Scattering2D
import numpy as np
import os
import random
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import torch.nn.functional as F

class WaveVIT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, scale=2):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.Scale = scale

        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)

        self.vit.patch_embed = nn.Identity()

        self.cls_token = self.vit.cls_token
        self.pos_embed = self.vit.pos_embed
        self.pos_drop = self.vit.pos_drop
        self.blocks = self.vit.blocks
        self.norm = self.vit.norm
        self.head = self.vit.head

        self.scatter = Scattering2D(J=self.Scale, shape=(img_size, img_size))

        self.hidden_linear = nn.Linear(243, embed_dim)  # 243 is c*d 
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(243, 243, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(243, 243, kernel_size=4, stride=2, padding=1)
        )
        self.projection = nn.Conv2d(embed_dim, embed_dim, kernel_size=patch_size, stride=patch_size)

    def WaveScatter(self, x):
        x = x.cpu().numpy()

        if x.ndim == 4 and x.shape[1] == 3:
            out = []
            for sample in x:
                # Sample shape: (3, H, W)
                scat_sample = self.scatter(sample)
                out.append(scat_sample)
            x = np.stack(out)
        else:
            raise ValueError("Expected input shape (B, 3, H, W)")
        
        return torch.from_numpy(x).float().to(self.device)

    def forward(self, x):
        x = self.WaveScatter(x) 
        b, c, d, h, w = x.shape
        x = x.view(b, c * d, h, w)  

        x = self.upsample(x)  
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.hidden_linear(x)  # (B, H, W, embed_dim)
        x = x.permute(0, 3, 1, 2)  # (B, embed_dim, H, W)

        # Patchify
        x = self.projection(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        x = x + self.pos_embed[:, :x.size(1)]
        x = self.pos_drop(x)

        x = self.blocks(x)
        x = self.norm(x)

        return x[:, 0]  # classify with CLS token

class SiameseReIDModel(nn.Module):
    def __init__(self, feature_extractor):
        super().__init__()
        self.encoder = feature_extractor
    
    def forward(self, anchor, positive, negative):
        anchor_feat = self.encoder(anchor)
        positive_feat = self.encoder(positive)
        negative_feat = self.encoder(negative)
        
        return anchor_feat, positive_feat, 

class TripletReIDDataset(Dataset):
    def __init__(self, triplet_paths, transform=None):
        """
        triplet_paths: list of (anchor_path, positive_path, negative_path)
        transform: torchvision.transforms pipeline
        """
        self.triplet_paths = triplet_paths
        self.transform = transform

    def __len__(self):
        return len(self.triplet_paths)

    def __getitem__(self, idx):
        a_path, p_path, n_path = self.triplet_paths[idx]

        a_img = Image.open(a_path).convert("RGB")
        p_img = Image.open(p_path).convert("RGB")
        n_img = Image.open(n_path).convert("RGB")

        if self.transform:
            a_img = self.transform(a_img)
            p_img = self.transform(p_img)
            n_img = self.transform(n_img)

        return a_img, p_img, n_img

def generate_triplets_from_market1501(data_dir, num_triplets=1000, mode='train'):

    assert mode in ['train', 'test'], "mode must be 'train' or 'test'"
    subdir = 'bounding_box_train' if mode == 'train' else 'bounding_box_test'
    image_dir = os.path.join(data_dir, subdir)
    pid_to_images = defaultdict(list)

    for fname in os.listdir(image_dir):
        if fname.endswith('.jpg'):
            pid = int(fname.split('_')[0])
            pid_to_images[pid].append(os.path.join(image_dir, fname))
        
    pids = list(pid_to_images.keys())
    triplets = []

    for _ in range(num_triplets):
        anchor_pid = random.choice(pids)
        positive_images = pid_to_images[anchor_pid]

        if len(positive_images) < 2:
            continue
        
        anchor_path, positive_path = random.sample(positive_images, 2)
        negative_pid = random.choice([pid for pid in pids if pid != anchor_pid])
        negative_pid = random.choice(pid_to_images[negative_pid])

        triplets.append((anchor_path, positive_path, negative_pid))
    
    return triplets



