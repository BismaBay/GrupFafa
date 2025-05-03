import os
import numpy as np
from sklearn.svm import SVC
import joblib

embedding_folder = "D:\SEMESTER 6\9. Desain Proyek\DATASET\Hasil Embadding Face"
embeddings = []
labels = []

# Ambil semua embedding dan label nama dari folder
for person_name in os.listdir(embedding_folder):
    person_folder = os.path.join(embedding_folder, person_name)

    if not os.path.isdir(person_folder):
        continue

    for file in os.listdir(person_folder):
        if file.endswith("_embedding.npy"):
            path = os.path.join(person_folder, file)
            emb = np.load(path)
            embeddings.append(emb)
            labels.append(person_name)

# Konversi ke array
X = np.array(embeddings)
y = np.array(labels)

# Train SVM classifier
clf = SVC(kernel='linear', probability=True)
clf.fit(X, y)

# Simpan model
model_path = "D:\SEMESTER 6\9. Desain Proyek\DATASET\svm_multiclass_model.pkl"
joblib.dump(clf, model_path)

print("Training selesai! Model disimpan.")
