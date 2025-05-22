import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from WaveVIT import SiameseReIDModel, WaveVIT

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SiameseReIDModel(WaveVIT())
model.load_state_dict(torch.load(r"D:\Tarsyer Work\recorder\WaveVIT\model_epoch_8\model_epoch_8.pt", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                          std=[0.229, 0.224, 0.225]),
])

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image.unsqueeze(0).to(device)  # shape: (1, 3, H, W)

img1_tensor = preprocess_image(r"D:\Tarsyer Work\recorder\WaveVIT\P19.jpg")
img2_tensor = preprocess_image(r"D:\Tarsyer Work\recorder\WaveVIT\P20.jpg")

with torch.no_grad():
    embedding1 = model(img1_tensor)  # shape: (1, D)
    embedding2 = model(img2_tensor)  # shape: (1, D)

    embedding1 = F.normalize(embedding1, p=2, dim=1)
    embedding2 = F.normalize(embedding2, p=2, dim=1)

    distance = torch.norm(embedding1 - embedding2, p=2).item()

    print(f"Distance between the two embeddings: {distance:.4f}")

    threshold = 0.028
    if distance < threshold:
        print("✅ Same identity (match)")
    else:
        print("❌ Different identities (no match)")
