from torch.utils.data import DataLoader
from WaveVIT import generate_triplets_from_market1501, TripletReIDDataset
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from PIL import Image
import random
from collections import defaultdict
from torch.utils.data.sampler import Sampler

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


def imshow(img_tensor, title=None):
    img = img_tensor.cpu().numpy().transpose((1, 2, 0)) 
    img = np.clip(img, 0, 1)  
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis('off')


if __name__ == "__main__":

    triplets = generate_triplets_from_market1501(
        data_dir=r"D:\Tarsyer Work\recorder\Market-1501-v15.09.15",
        num_triplets=10000,
        mode='train'
    )

    transform_train = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((256, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    test_transform = T.Compose([
        T.Resize((256, 128)),  
        T.ToTensor(),  
    ])

    # train_dataset = TripletReIDDataset(triplets, transform=transform_train)
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

    test_dataset = TripletReIDDataset(triplets, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    anchor_batch, positive_batch, negative_batch = next(iter(test_loader)) 

    plt.figure(figsize=(15, 5))
    for i in range(5):
        plt.subplot(3, 5, i + 1)
        imshow(anchor_batch[i], title='Anchor')

        plt.subplot(3, 5, i + 6)
        imshow(positive_batch[i], title='Positive')

        plt.subplot(3, 5, i + 11)
        imshow(negative_batch[i], title='Negative')

    plt.tight_layout()
    plt.show()