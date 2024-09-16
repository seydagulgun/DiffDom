import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import copy

# MNIST verileri için dönüştürme işlemi (her bir görüntüyü tensora dönüştürür)
transform = transforms.ToTensor()

# MNIST veri setini indir ve yükle (train verisi)
dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

# Müşteri (client) sayısı belirleniyor
NUM_CLIENTS = 5

# Veri seti 5 parçaya bölünüyor (5 farklı müşteri için)
client_datasets = random_split(dataset, [len(dataset) // NUM_CLIENTS] * NUM_CLIENTS)

# Her müşteri için veriyi yükleyecek bir DataLoader oluşturuluyor
client_loaders = [DataLoader(ds, batch_size=4, shuffle=True) for ds in client_datasets]

# Test veri seti tüm müşteriler için aynı kalıyor (train değil, test verisi)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Basit bir CNN (Convolutional Neural Network) modeli tanımlanıyor
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # İlk convolution katmanı (1 giriş kanalı, 16 çıkış filtresi, 3x3 kernel)
        self.conv1 = nn.Conv2d(1, 16, 3)
        # İkinci convolution katmanı (16 giriş kanalı, 32 çıkış filtresi, 3x3 kernel)
        self.conv2 = nn.Conv2d(16, 32, 3)
        # Tam bağlantılı (fully connected) katman
        self.fc1 = nn.Linear(32 * 12 * 12, 64)
        # Çıkış katmanı (10 sınıf için)
        self.fc2 = nn.Linear(64, 10)

    # İleri yayılım (forward propagation) metodu
    def forward(self, x):
        # İlk convolution katmanına ReLU aktivasyon fonksiyonu uygulanıyor
        x = torch.relu(self.conv1(x))
        # İkinci convolution katmanına ReLU uygulanıyor
        x = torch.relu(self.conv2(x))
        # Max pooling uygulanıyor (2x2 kernel boyutunda)
        x = torch.max_pool2d(x, 2)
        # Veriler düzleştiriliyor (flatten)
        x = x.view(-1, 32 * 12 * 12)
        # Tam bağlantılı katmana ReLU uygulanıyor
        x = torch.relu(self.fc1(x))
        # Son çıkış katmanına veri gönderiliyor
        x = self.fc2(x)
        return x

# Küresel modelin başlangıçta tanımlanması
global_model = CNN()

# Kayıp fonksiyonu ve optimizasyon algoritması (kriter olarak çapraz entropi)
criterion = nn.CrossEntropyLoss()

# Müşteri tarafındaki eğitim fonksiyonu
def client_update(client_model, train_loader, optimizer, epochs=1):
    # Modeli eğitim moduna geçir
    client_model.train()
    for _ in range(epochs):
        for data, target in train_loader:
            # Optimizasyonu sıfırla
            optimizer.zero_grad()
            # Model ile tahmin yap
            output = client_model(data)
            # Kayıp hesapla
            loss = criterion(output, target)
            # Geri yayılım (backpropagation) ile ağırlıkları güncelle
            loss.backward()
            # Optimizasyon adımı gerçekleştir
            optimizer.step()

# Sunucu tarafındaki model birleştirme (Federated Averaging) fonksiyonu
def server_aggregate(global_model, client_models):
    # Küresel modelin ağırlıkları alınıyor
    global_dict = global_model.state_dict()
    # Her bir ağırlık için, tüm müşteri modellerindeki ağırlıkların ortalaması alınıyor
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i].state_dict()[k] for i in range(NUM_CLIENTS)], 0).mean(0)
    # Küresel modelin ağırlıkları güncelleniyor
    global_model.load_state_dict(global_dict)

# Modelin doğruluğunu test eden fonksiyon
def test_model(model, test_loader):
    # Model test moduna alınıyor
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # Test verisi ile model tahmini yapılıyor
            output = model(data)
            # Tahmin edilen sınıf bulunuyor
            pred = output.argmax(dim=1, keepdim=True)
            # Doğru tahmin edilenlerin sayısı hesaplanıyor
            correct += pred.eq(target.view_as(pred)).sum().item()
    # Doğruluk oranı hesaplanıyor ve yazdırılıyor
    accuracy = correct / len(test_loader.dataset)
    print(f'Accuracy: {accuracy:.4f}')

# Ana federated learning fonksiyonu
def federated_learning(global_model, client_loaders, test_loader, rounds=5):
    # Her tur için döngü
    for round_num in range(rounds):
        print(f'\nRound {round_num+1}/{rounds}')

        # Adım 1: Her müşteri kendi verisi üzerinde eğitiliyor
        client_models = [copy.deepcopy(global_model) for _ in range(NUM_CLIENTS)]
        optimizers = [optim.Adam(model.parameters(), lr=0.001) for model in client_models]

        # Her müşteri modeli kendi verisi üzerinde eğitiliyor
        for i in range(NUM_CLIENTS):
            client_update(client_models[i], client_loaders[i], optimizers[i])

        # Adım 2: Sunucu modelleri birleştiriyor (Federated Averaging)
        server_aggregate(global_model, client_models)

        # Adım 3: Küresel model test ediliyor
        test_model(global_model, test_loader)

# Federated learning işlemi başlatılıyor
federated_learning(global_model, client_loaders, test_loader, rounds=5)
