from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import random
import csv
import os
import PyPDF2  # PDF işlemleri için
import re

app = Flask(__name__)

# Klasör yolları
DATA_DIR = 'data'
UPLOAD_DIR = 'uploads'
TEMPLATES_DIR = 'templates'

# Gereken klasörleri oluştur
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Veri Yolu
file_path = os.path.join(DATA_DIR, 'veri_seti_supervisor.csv')
log_file = os.path.join(DATA_DIR, 'evrak_kayitlari.csv')

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
    # Küçük harfe dönüştürme
    text = text.lower()
    # Gereksiz karakterleri kaldırma (Türkçe karakterleri koru)
    text = re.sub(r'[^a-zA-ZçÇğĞıİöÖşŞüÜ\s]', '', text)
    # Stop words çıkarma
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

# 4. Model Eğitimi
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# 5. Performans Kontrolü
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

# 6. Dağıtım Fonksiyonu
feature_names = np.array(vectorizer.get_feature_names_out())
log_probs = model.feature_log_prob_

# Log dosyası başlatma
def initialize_log_file():
    if not os.path.exists(log_file):
        with open(log_file, mode='w', newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file)
            writer.writerow(['Belge ID', 'Atanan Birim', 'Önemli Kelimeler', 'Güven Skoru'])

initialize_log_file()

# Analiz ve Dağıtım Fonksiyonu
def analyze_and_assign(document, doc_id):
    document = preprocess_text(document)
    vec_doc = vectorizer.transform([document])
    probabilities = model.predict_proba(vec_doc)[0]
    random_choice = random.choices(model.classes_, weights=probabilities, k=1)[0]
    explanation = {}

    class_index = list(model.classes_).index(random_choice)
    top_indices = log_probs[class_index].argsort()[-10:][::-1]
    important_words = feature_names[top_indices]
    explanation['important_words'] = important_words.tolist()
    explanation['confidence'] = probabilities[class_index]

    with open(log_file, mode='a', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow([doc_id, random_choice, ', '.join(important_words), explanation['confidence']])

    return random_choice, explanation

# PDF İşleme Fonksiyonu
def pdf_to_text(file_path):
    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print("PyPDF2 Hatası:", e)
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['file']
    if file:
        # PDF'i işle
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        file.save(file_path)
        content = pdf_to_text(file_path)

        doc_id = random.randint(1000, 9999)
        assigned_unit, explanation = analyze_and_assign(content, doc_id)
        return render_template('result.html', 
        assigned_unit=assigned_unit, explanation=explanation)
    return "Dosya yüklenmedi!"

if __name__ == '__main__':
    app.run(debug=True)
