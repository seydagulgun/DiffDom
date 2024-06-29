import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10

# CIFAR-10 veri setini yükle
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Test setinden rastgele 5000 görüntü seç
indices = np.random.choice((x_test.shape[0]), 5000, replace=False)
subset = x_test[indices]

print(subset.shape)

# Bu 5000 görüntü içinden rastgele bir görüntü seç
random_index_subset = np.random.randint(subset.shape[0])
selected_image = subset[random_index_subset]

# Seçilen görüntüyü göster
plt.imshow(selected_image)
plt.title("Seçilen Görüntü")

# Benzer görüntüleri bulma fonksiyonu
def find_similar_images(image, dataset, num_images=11):
    # Görüntüler arası farkların karesini hesapla
    differences = dataset - image
    print(dataset.shape)
    print(image.shape)
    print(differences.shape)
    squared_differences = differences ** 2
    print(squared_differences.shape)
    # Her bir görüntü için farkların toplamını hesapla (öklid mesafesi)
    distances = np.sqrt(squared_differences.sum(axis=(1,2,3)))
    print(distances.shape)
    # En küçük mesafelere sahip görüntülerin indekslerini bul
    closest_indices = np.argsort(distances)[1:num_images]
    return dataset[closest_indices]

# Benzer görüntüleri bul
similar_images = find_similar_images(selected_image, subset)

# Benzer görüntüleri göster
plt.figure(figsize=(15, 5))
for i, similar_image in enumerate(similar_images):
    plt.subplot(2, 5, i+1)
    plt.imshow(similar_image)
    plt.title(f"Benzer {i+1}")
    plt.axis('off')
plt.tight_layout()