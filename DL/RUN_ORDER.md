# Run Order Guide

## Urutan Menjalankan Proyek

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Struktur File (Urutan Alur NLP Pipeline)

```
DL/
├── src/
│   ├── __init__.py
│   ├── data_processing.py     # 1. Load, Preprocessing, Split, & TF-IDF
│   └── training.py            # 2. Training model
├── evaluasi.py                # 3. Evaluasi model
├── aplikasi.py                # 4. Web application
├── data/
│   └── dataset.csv
├── models/                    # Untuk menyimpan model trained
├── templates/                 # HTML templates
└── static/                    # CSS & JS files
```

### 3. Urutan Eksekusi (Alur NLP Pipeline)

**Urutan File sesuai alur:**
1. `src/data_processing.py` - Load dataset, Preprocessing teks (case folding, tokenization, stopword removal, stemming), Split data, & Ekstraksi fitur TF-IDF
2. `src/training.py` - Training model klasifikasi
3. `evaluasi.py` - Evaluasi model dan save hasil
4. `aplikasi.py` - Web application untuk klasifikasi real-time

**Semua file di src/ bersifat independen dan dapat di-import dalam urutan apapun.**

### 4. Cara Menjalankan

**Option A: Training & Evaluasi (CLI) - RECOMMENDED**
```bash
python evaluasi.py
```
Ini akan menampilkan semua langkah secara jelas:
1. Load dataset dari `data/raw/dataset.csv`
2. Preprocessing teks (case folding, tokenization, stopword removal, stemming)
3. Split data (train/test)
4. Ekstraksi fitur TF-IDF
5. Training model
6. Evaluasi model
7. Save model ke `models/`

**Option B: Web Application**
```bash
python aplikasi.py
```
Lalu buka browser: `http://localhost:5000`

Di web interface:
1. Klik "Train Model" untuk training (semua langkah otomatis)
2. Klik "Load Model" untuk load model yang sudah disimpan
3. Masukkan title dan description untuk klasifikasi

### 5. Alur Kerja NLP Pipeline

```
Dataset (CSV)
    ↓
Data Processing (src/data_processing.py)
    - Load Dataset
    - Case Folding
    - Remove Special Characters
    - Tokenization
    - Stopword Removal
    - Stemming
    - Train/Test Split
    - TF-IDF Extraction
    ↓
Training (src/training.py)
    - Model Training
    ↓
Evaluasi (evaluasi.py)
    - Model Evaluation
    - Save Model
    ↓
Aplikasi (aplikasi.py)
    - Web Interface
    - Real-time Classification
```

### 6. File Utama untuk Dijalankan

- **evaluasi.py** - Script training & evaluasi model (CLI)
- **aplikasi.py** - Web application Flask

### 7. Dependencies Eksternal

- nltk (untuk tokenization dan stopwords)
- sklearn (untuk TF-IDF dan model ML)
- pandas (untuk data handling)
- flask (untuk web app)

NLTK data akan di-download otomatis saat pertama kali dijalankan.
