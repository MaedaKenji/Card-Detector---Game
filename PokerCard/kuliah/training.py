import cv2
import numpy as np
from skimage.feature import hog
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def extract_features(image):
    # Konversi gambar ke grayscale jika belum
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize gambar ke ukuran tetap
    image = cv2.resize(image, (64, 64))
    
    # Ekstrak fitur HOG
    features, _ = hog(image, orientations=8, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=True)
    return features

# Fungsi untuk memuat dan mempersiapkan dataset
def load_dataset(data_path):
    images = []
    labels = []
    # Kode untuk memuat gambar dan label dari data_path
    # ...
    return images, labels

# Persiapkan dataset
data_path = "dataset"
images, labels = load_dataset(data_path)

# Ekstrak fitur dari semua gambar
features = [extract_features(img) for img in images]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Normalisasi fitur
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM model
clf = svm.SVC(kernel='rbf', C=1)
clf.fit(X_train_scaled, y_train)

# Evaluasi model
accuracy = clf.score(X_test_scaled, y_test)
print(f"Model accuracy: {accuracy}")

# Fungsi untuk prediksi
def predict_card(image):
    features = extract_features(image)
    features_scaled = scaler.transform([features])
    return clf.predict(features_scaled)[0]

# Contoh penggunaan
test_image = cv2.imread('path/to/test/image.jpg')
prediction = predict_card(test_image)
print(f"Predicted card: {prediction}")