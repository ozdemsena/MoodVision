import torch
from torchvision import transforms
from PIL import Image
from model import EmotionCNN

# Duygu etiketleri
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Görüntü dönüşümleri
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Modeli yükle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionCNN(num_classes=7).to(device)
model.load_state_dict(torch.load("emotion_cnn_augmented.pth", map_location=device))
model.eval()

# Tahmin fonksiyonu
def predict_image(image_path):
    img = Image.open(image_path).convert("L")  # grayscale
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
    
    return emotion_labels[predicted.item()]

if __name__ == "__main__":
    test_img = "face.jpg"  # Kendi yüz fotoğrafını buraya koy
    result = predict_image(test_img)
    print("Tahmin edilen duygu:", result)
