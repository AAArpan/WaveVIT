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

        anchor_positive_dist = pairwise_dist.unsqueeze(2)
        anchor_negative_dist = pairwise_dist.unsqueeze(1)

        triplet_loss = anchor_positive_dist - anchor_negative_dist + self.margin
        mask = anchor_positive_mask.unsqueeze(2) & anchor_negative_mask.unsqueeze(1)
        triplet_loss = triplet_loss * mask

        triplet_loss = F.relu(triplet_loss)

        valid_triplets = triplet_loss > 1e-16
        num_positive_triplets = valid_triplets.sum().float()

        loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)

        # For logging: average positive and negative distances
        pos_dist = (pairwise_dist * anchor_positive_mask).sum() / (anchor_positive_mask.sum() + 1e-16)
        neg_dist = (pairwise_dist * anchor_negative_mask).sum() / (anchor_negative_mask.sum() + 1e-16)

        return loss, pos_dist, neg_dist

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



