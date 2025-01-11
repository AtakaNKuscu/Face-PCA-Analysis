import os
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.io import imsave
from sklearn.decomposition import PCA
import cv2

# Function to load .pgm images from a given directory
def load_pgm_images_from_directory(directory_path):
    image_list = []
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.pgm'):
            file_path = os.path.join(directory_path, file_name)
            image = imread(file_path, as_gray=True)
            image_list.append(image.flatten())  # Flatten to 1D array
    return np.array(image_list)

# Update these paths to your directories containing .pgm files
directory1 = "C:/Users/msı/OneDrive/Masaüstü/machine learning/hw3/face1"
directory2 = "C:/Users/msı/OneDrive/Masaüstü/machine learning/hw3/face2"

# Load images from both directories
images1 = load_pgm_images_from_directory(directory1)
images2 = load_pgm_images_from_directory(directory2)
combined_images = np.vstack((images1, images2))

# 3. PCA Uygulaması
pca = PCA(n_components=5)  # İlk 5 temel bileşeni (öz yüz) seçiyoruz
combined_images_pca = pca.fit_transform(combined_images)

# Kovaryans Matrisi hesaplama
cov_matrix = np.cov(combined_images.T)
print("Hesaplanan Kovaryans Matrisi:\n", cov_matrix)

# 4. Özdeğerler ve Özvektörler
eigenvalues = pca.explained_variance_
eigenvectors = pca.components_

# 5. En büyük 10 özdeğere karşılık gelen özvektörleri al
top_10_eigenvectors = eigenvectors[:10]  # İlk 10 özvektörü alıyoruz

# 6. İlk 5 özvektörü (öz yüzleri) seç
top_5_eigenvectors = top_10_eigenvectors[:5]

# 7: Visualize and save the first 5 eigenfaces
for i in range(5):
    eigenface = top_10_eigenvectors[i].reshape((192, 168))  # Reshape to image dimensions
    
    # Normalize the eigenface to the range [0, 255] for image saving
    eigenface_normalized = np.uint8((eigenface - np.min(eigenface)) / np.ptp(eigenface) * 255)
    
    # Show the eigenface
    plt.imshow(eigenface, cmap='gray')
    plt.axis('off')
    plt.title(f"Eigenface {i+1}")
    plt.show()
    
    # Save the image
    imsave(f"eigenface_{i+1}.png", eigenface_normalized)

# 8. PCA Uzayına Projeksiyon ve Yeniden Yapılandırma
def reconstruct_face_from_pca(projection, mean_face, eigenfaces):
    """
    PCA projeksiyonunu kullanarak yüzü yeniden yapılandırır.
    """
    reconstructed_face = mean_face + np.dot(projection, eigenfaces)
    return reconstructed_face

# PCA ile projeksiyon sonuçlarını al
projected_images = combined_images_pca  # PCA sonucu yüzlerin projeksiyonu

# 9. Projeksiyonları tersine çevirerek yüzleri yeniden oluşturma
mean_face = np.mean(combined_images, axis=0)  # Orta yüzü hesapla
reconstructed_faces = []

for i, proj in enumerate(projected_images):
    reconstructed_face = reconstruct_face_from_pca(proj, mean_face, eigenvectors[:5])  # Yeniden yapılandırma
    reconstructed_faces.append(reconstructed_face)

# 10. Yeniden yapılandırılmış yüzleri 192x168 boyutunda kaydetme
for i, rec_face in enumerate(reconstructed_faces):
    rec_face_image = rec_face.reshape(192, 168)  # Orijinal boyutlara dönüştür
    rec_face_image = np.uint8(np.abs(rec_face_image))  # Pozitif değerler için
    cv2.imwrite(f"reconstructed_face_{i+1}.png", rec_face_image)


# 11. Ortalama yüzü kaydetme (görsel olarak)
mean_face_image = mean_face.reshape(192, 168)  # Ortalama yüzü yeniden boyutlandır
mean_face_image = np.uint8(np.abs(mean_face_image))  # Pozitif değerler için
cv2.imwrite("mean_face.png", mean_face_image)  # Ortalama yüzü kaydetme

# 12. Euclidean mesafesini hesaplama (yeniden yapılandırılmış yüz ile orijinal yüz arasındaki fark)
def euclidean_distance(face1, face2):
    """
    İki yüz arasındaki Euclidean mesafesini hesaplar.
    """
    return np.linalg.norm(face1 - face2)

# 13. Orijinal yüzlerle yeniden yapılandırılmış yüzler arasındaki mesafeleri hesaplama
original_faces = combined_images  # Orijinal yüzler (tek boyutlu vektörler)
distances = []

# Mesafeleri hesapla
for i in range(len(original_faces)):
    distance = euclidean_distance(original_faces[i], reconstructed_faces[i])
    distances.append(distance)

# 14. Yüzleri ve Euclidean mesafelerini aynı plotta gösterme
fig, axes = plt.subplots(nrows=10, ncols=2, figsize=(10, 20))

# 15. Yüzleri ve Euclidean mesafelerini her bir yüz için ayrı plot'ta gösterme
for i in range(10):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))  # 1 satır, 2 sütun

    # Orijinal ve yeniden yapılandırılmış yüzleri görüntüle
    original_face_image = original_faces[i].reshape(192, 168)
    reconstructed_face_image = reconstructed_faces[i].reshape(192, 168)
    
    # Orijinal yüzü göster
    axes[0].imshow(original_face_image, cmap='gray')
    axes[0].set_title(f"Original Face {i+1}")
    axes[0].axis('off')
    
    # Yeniden yapılandırılmış yüzü göster
    axes[1].imshow(reconstructed_face_image, cmap='gray')
    axes[1].set_title(f"Reconstructed Face {i+1}")
    axes[1].axis('off')
    
    # Euclidean mesafesini yaz
    axes[1].text(-0.4, -0.1, f"Euclidean Distances: {distances[i]:.4f}", ha='center', va='center', transform=axes[1].transAxes)
    
    # Plot'u göster
    plt.tight_layout()
    plt.show()

# 16. Raporlama bilgisi
print("Yüz görüntüleri PCA uzayına projekte edildi ve yeniden yapılandırılıp kaydedildi.")