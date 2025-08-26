import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import FER2013Dataset
from model import EmotionCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training on:", device)

# ===== Veri dönüşümleri (Data Augmentation ekledik) =====
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.RandomHorizontalFlip(),  # Yatay çevirme → veri çeşitliliği
    transforms.RandomRotation(10),      # ±10 derece döndürme → veri çeşitliliği
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ===== Dataset ve DataLoader =====
train_dataset = FER2013Dataset(csv_file="fer2013.csv", usage="Training", transform=transform)
val_dataset   = FER2013Dataset(csv_file="fer2013.csv", usage="PublicTest", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=64)

# ===== Model =====
model = EmotionCNN(num_classes=7).to(device)

# ===== Dropout =====
# Model dosyamızda dropout zaten var. Eğer artırmak istersek model.py'de:
# self.dropout = nn.Dropout(0.6)  # eskiden 0.5 idi → overfitting azaltır

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ===== Eğitim döngüsü =====
num_epochs = 3  # Daha uzun eğitim, overfitting'i azaltmak için augmentation ile birlikte
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    # ===== Validation =====
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.2f}%")

# ===== Model kaydetme =====
torch.save(model.state_dict(), "emotion_cnn_augmented.pth")
print("Model saved as emotion_cnn_augmented.pth")
