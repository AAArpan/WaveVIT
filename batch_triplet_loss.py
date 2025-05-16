import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data.sampler import Sampler
from collections import defaultdict
import random
import numpy as np
from PIL import Image

class BatchAllTripletLoss(nn.Module):
    def __init__(self, margin=0.3, squared=False):
        super(BatchAllTripletLoss, self).__init__()
        self.margin = margin
        self.squared = squared

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: Tensor of shape (B, D) where D is embedding dimension.
            labels: Tensor of shape (B,) with class indices.
        Returns:
            triplet loss (scalar)
        """
        pairwise_dist = self._pairwise_distance(embeddings)

        anchor_positive_mask = self._get_anchor_positive_mask(labels)
        anchor_negative_mask = self._get_anchor_negative_mask(labels)

        # Compute triplet loss
        anchor_positive_dist = pairwise_dist.unsqueeze(2)  # (B, B, 1)
        anchor_negative_dist = pairwise_dist.unsqueeze(1)  # (B, 1, B)

        triplet_loss = anchor_positive_dist - anchor_negative_dist + self.margin

        mask = anchor_positive_mask.unsqueeze(2) & anchor_negative_mask.unsqueeze(1)
        triplet_loss = triplet_loss * mask

        triplet_loss = F.relu(triplet_loss)

        valid_triplets = triplet_loss > 1e-16
        num_positive_triplets = valid_triplets.sum().float()

        triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)

        return triplet_loss

    def _pairwise_distance(self, embeddings):
        """Efficient pairwise Euclidean distance computation"""
        dot_product = torch.matmul(embeddings, embeddings.t())
        square = torch.diagonal(dot_product, 0)
        dist = square.unsqueeze(0) - 2 * dot_product + square.unsqueeze(1)
        dist = F.relu(dist)  

        if not self.squared:
            dist = torch.sqrt(dist + 1e-16)
        return dist

    def _get_anchor_positive_mask(self, labels):
        """Mask for valid anchor-positive pairs"""
        labels = labels.unsqueeze(0)
        return labels == labels.t()

    def _get_anchor_negative_mask(self, labels):
        """Mask for valid anchor-negative pairs"""
        labels = labels.unsqueeze(0)
        return labels != labels.t()

class Market1501Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.pid_to_label = {}
        self.mode = mode

        subdir = 'bounding_box_train' if mode == 'train' else 'bounding_box_test'
        data_dir = os.path.join(root_dir, subdir)
        pid_set = set()

        for fname in os.listdir(data_dir):
            if fname.endswith('.jpg'):
                pid = int(fname.split('_')[0])
                self.image_paths.append(os.path.join(data_dir, fname))
                self.labels.append(pid)
                pid_set.add(pid)

        # Convert original PIDs to continuous 0-N labels
        pid_list = sorted(pid_set)
        self.pid_to_label = {pid: idx for idx, pid in enumerate(pid_list)}
        self.labels = [self.pid_to_label[pid] for pid in self.labels]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

#Group data by identity (PID)
class RandomIdentitySampler(Sampler):
    def __init__(self, labels, num_instances=4, batch_size=32):
        self.labels = labels
        self.num_instances = num_instances
        self.batch_size = batch_size
        self.index_dic = defaultdict(list)
        for idx, label in enumerate(labels):
            self.index_dic[label].append(idx)
        self.pids = list(self.index_dic.keys())
        self.num_identities = batch_size // num_instances

    def __iter__(self):
        indices = []
        random.shuffle(self.pids)
        for pid in self.pids:
            idxs = self.index_dic[pid]
            if len(idxs) >= self.num_instances:
                selected = random.sample(idxs, self.num_instances)
            else:
                selected = np.random.choice(idxs, self.num_instances, replace=True)
            indices.extend(selected)
        return iter(indices)

    def __len__(self):
        return len(self.labels)

