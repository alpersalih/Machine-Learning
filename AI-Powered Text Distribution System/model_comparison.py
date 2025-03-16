import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import re
import os

# Klasör yolları
DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)

# Veri yolu
file_path = os.path.join(DATA_DIR, 'veri_seti_supervisor.csv')

# 1. Veri Yükleme
data = pd.read_csv(file_path, encoding='utf-8-sig')

# 2. Metin Temizleme ve Önişleme
stop_words = [
    've', 'bir', 'bu', 'şu', 'için', 'olan', 'ile', 'de', 'da', 'ki', 
    'ama', 'ancak', 'fakat', 'çünkü', 'veya', 'gibi', 'ile', 'eğer', 
    'hem', 'ne', 'ya', 'ise', 'kadar', 'sonra', 'önce', 'çok', 'az', 
    'her', 'bazı', 'diğer', 'daha', 'hemen', 'bütün', 'sadece', 'artık',
    'neden', 'nasıl', 'şimdi', 'yani', 'zaten', 'en', 'böyle', 'işte']

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-ZçÇğĞıİöÖşŞüÜ\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Metinlerin temizlenmesi
data['Belge İçeriği'] = data['Belge İçeriği'].apply(preprocess_text)

# 3. Özellik Çıkarma (TF-IDF)
X = data['Belge İçeriği']
y = data['Birim']
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X_tfidf = vectorizer.fit_transform(X)

# 4. Eğitim ve Test Bölünmesi
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# 5. Modellerin Tanımlanması
models = {
    'MultinomialNB': MultinomialNB(),
    'SVM': SVC(kernel='linear'),
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'RandomForest': RandomForestClassifier(n_estimators=100),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

# 6. Model Eğitimi ve Testi
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = {
        'Accuracy': acc,
        'Report': classification_report(y_test, y_pred)
    }

# Sonuçların Gösterimi
for model_name, result in results.items():
    print(f"Model: {model_name}")
    print(f"Accuracy: {result['Accuracy']:.2f}")
    print("Classification Report:")
    print(result['Report'])
    print("----------------------------------------")

# Algoritmaların doğruluk değerlerine göre sıralanması
sorted_results = sorted(results.items(), key=lambda x: x[1]['Accuracy'], reverse=True)
print("Algoritmaların Doğruluk Değerleri (Sıralı):")
for model_name, result in sorted_results:
    print(f"{model_name}: {result['Accuracy']:.2f}")
