import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

# 1. Veri kümesinin yüklenmesi ve dönüştürülmesi
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #ilki her kanalın ortalama değeri, ikincisi ise her kanalın standart sapması

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2) #num_workers???

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 2. Modelin tanımlanması
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # İlk konvolüsyon katmanı
        self.conv1 = nn.Conv2d(3, 6, 5)
        # Girdi: 3x32x32
        # 5x5 filtre, padding=0 ve stride=1 varsayılan değerler
        # Çıkış: 6x28x28 (çıkış boyutu: (32 - 5 + 1) x (32 - 5 + 1) = 28 x 28)

        # Max pooling katmanı
        self.pool = nn.MaxPool2d(2, 2)
        # Girdi: 6x28x28
        # 2x2 havuzlama ve stride=2
        # Çıkış: 6x14x14 (çıkış boyutu: 28 / 2 x 28 / 2 = 14 x 14)

        # İkinci konvolüsyon katmanı
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Girdi: 6x14x14
        # 5x5 filtre, padding=0 ve stride=1 varsayılan değerler
        # Çıkış: 16x10x10 (çıkış boyutu: (14 - 5 + 1) x (14 - 5 + 1) = 10 x 10)

        # İkinci Max pooling katmanı
        self.pool = nn.MaxPool2d(2, 2)
        # Girdi: 16x10x10
        # 2x2 havuzlama ve stride=2
        # Çıkış: 16x5x5 (çıkış boyutu: 10 / 2 x 10 / 2 = 5 x 5)

        # İlk tam bağlantılı katman (fully connected layer)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # Girdi: 16x5x5 (özellik haritası düzleştirilir: 16 * 5 * 5 = 400)
        # Çıkış: 120 (tam bağlantılı katmandan 120 nöron)

        # İkinci tam bağlantılı katman
        self.fc2 = nn.Linear(120, 84)
        # Girdi: 120
        # Çıkış: 84 (tam bağlantılı katmandan 84 nöron)

        # Üçüncü tam bağlantılı katman
        self.fc3 = nn.Linear(84, 10)
        # Girdi: 84
        # Çıkış: 10 (CIFAR-10 veri kümesindeki 10 sınıf için 10 nöron)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        # Konvolüsyon 1 ve ReLU sonrası: 6x28x28
        # Pooling sonrası: 6x14x14
        x = self.pool(F.relu(self.conv2(x)))
        # Konvolüsyon 2 ve ReLU sonrası: 16x10x10
        # Pooling sonrası: 16x5x5
        x = x.view(-1, 16 * 5 * 5)
        # Düzleştirme (flattening): 16*5*5 = 400
        x = F.relu(self.fc1(x))
        # İlk tam bağlantılı katman ve ReLU sonrası: 120
        x = F.relu(self.fc2(x))
        # İkinci tam bağlantılı katman ve ReLU sonrası: 84
        x = self.fc3(x)
        # Üçüncü tam bağlantılı katman sonrası: 10 (sınıflar)
        return x


net = Net()

# 3. Kayıp fonksiyonu ve optimizasyonun tanımlanması
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) #momentum ???

# 4. Modelin eğitilmesi
for epoch in range(2):  # 2 epoch üzerinden eğitim yapacağız
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0): #Eğitim veri yükleyicisini kullanarak verileri iteratif olarak çeker.
        inputs, labels = data #verileri ve etiketleri ayırır

        optimizer.zero_grad() #Gradientlerin sıfırlanması.

        outputs = net(inputs) #modelin ileri yayılımı
        loss = criterion(outputs, labels) #kayıp fonksiyonu kullanarak kaybın hesaplanması
        loss.backward() #geri yayılımla gradientlerin hesaplanması (pytorch bunu otomatik yapar)
        optimizer.step() #ağırlıkların güncellenmesi

        running_loss += loss.item() #kümülatif kaybı günceller
        if i % 2500 == 2499:    # Her 2500 adımda bir yazdır
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Eğitim tamamlandı')

# 5. Modelin test edilmesi
correct = 0
total = 0 #doğru tahminlerin ve toplam örneklerin sayısını sıfırlar
with torch.no_grad(): #Gradientlerin hesaplanmamasını engeller testte gerek yok
    for data in testloader: #Test veri yükleyicisini kullanarak verileri iteratif olarak çeker.
        images, labels = data #verileri ve etiketleri ayırır
        outputs = net(images) #modelin ileri yayılımı
        _, predicted = torch.max(outputs.data, 1) #tahmin edilen etiketleri bulur
        total += labels.size(0) #toplam örnek sayısını günceller
        correct += (predicted == labels).sum().item() #doğru tahminlerin sayısını günceller

print(f'10000 test görüntüsü üzerinden doğruluk: {100 * correct / total:.2f}%')
