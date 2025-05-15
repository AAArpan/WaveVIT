from torch.utils.data import DataLoader
from WaveVIT import generate_triplets_from_market1501, TripletReIDDataset
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import torch

def imshow(img_tensor, title=None):
    img = img_tensor.cpu().numpy().transpose((1, 2, 0)) 
    img = np.clip(img, 0, 1)  
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis('off')

triplets = generate_triplets_from_market1501(
    data_dir=r"D:\Tarsyer Work\recorder\Market-1501-v15.09.15",
    num_triplets=10000,
    mode='train'
)

transform_train = T.Compose([
    T.Resize((256, 128)),
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