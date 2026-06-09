# Implementasi K-Means Clustering untuk Segmentasi Tingkat Kemiskinan Provinsi di Indonesia

Proyek ini mengimplementasikan klastering K-Means untuk segmentasi tingkat kemiskinan provinsi Indonesia berdasarkan indikator sosial ekonomi. Proyek menggunakan KNIME sebagai tool data mining utama, dan Python Flask untuk dashboard web interaktif.

Dataset yang digunakan mencakup tiga indikator asli provinsi berikut:
- Nama Provinsi
- Tingkat Pengangguran Terbuka (Agustus 2025)
- Rata-rata Lama Sekolah (2025)
- Garis Kemiskinan Per Kapita (rata-rata Semester 2 Perkotaan dan Perdesaan)

Klastering dan visualisasi hanya menggunakan ketiga fitur utama tersebut.

## Struktur Folder

ML/
└── Project/
    ├── app/                  # Aplikasi web Flask dan script pembuat dataset
    │   ├── app.py           # File utama Flask (auto-generate dataset, train model)
    │   ├── templates/       # HTML templates
    │   └── static/          # File statis (CSS, JS, assets)
    ├── data/                 # Dataset mentah dan file hasil preprocessing
    ├── Docs/                 # Dokumentasi dan dataset mentah BPS
    │   ├── dataset/         # Dataset mentah BPS (pengangguran, lama sekolah, kemiskinan)
    │   ├── README.md        # File ini
    │   └── Laporan_Implementasi.md
    ├── knime/                # Deskripsi workflow KNIME dan dokumentasi node
    ├── output/               # Hasil klastering dan visualisasi Elbow
    └── requirements.txt      # Dependencies Python

## Isi Folder

- `app/app.py` : file utama Flask yang menggabungkan semua fungsi ML (load dataset, preprocessing, K-Means, elbow method, visualisasi, dan prediction)
- `app/templates/` : berisi template HTML untuk dashboard web
- `app/static/` : berisi file CSS, JS, dan assets statis
- `data/` : berisi dataset mentah dari `Docs/dataset` dan file `indonesia_sosial_ekonomi_2025.csv` hasil pemrosesan
- `Docs/dataset/` : berisi dataset mentah BPS (pengangguran, lama sekolah, garis kemiskinan) yang dipakai untuk membangun hasil klastering
- `output/` : berisi file `cluster_results.csv`, `cluster_summary.csv`, `elbow_plot.png`, dan model files (`.pkl`)
- `knime/` : berisi panduan workflow KNIME, node, pengaturan, dan visualisasi

## Instalasi

1. Buka terminal di folder `ML/Project`.
2. Buat virtual environment (opsional):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3. Install dependensi:
   ```powershell
   pip install -r requirements.txt
   ```

## Menjalankan Dashboard Flask

Cukup jalankan satu perintah:

```powershell
python app/app.py
```

Saat pertama dijalankan, `app.py` otomatis membaca dataset mentah dari `Docs/dataset/`, membangun fitur, melatih K-Means, dan menyimpan output berikut bila belum ada:

- `output/cluster_results.csv` - hasil klastering lengkap per provinsi
- `output/cluster_summary.csv` - ringkasan statistik per cluster
- `output/elbow_plot.png` - visualisasi metode elbow
- `output/scaler.pkl` - model StandardScaler
- `output/kmeans_model.pkl` - model K-Means
- `output/cluster_name_map.pkl` - mapping label cluster ke nama
- `data/indonesia_sosial_ekonomi_2025.csv` - dataset gabungan hasil preprocessing

Kalau ingin membangun ulang dari awal, hapus isi folder `output/` lalu jalankan lagi.

Buka `http://localhost:5000` di browser. Aplikasi dashboard menampilkan:
- Ringkasan klaster (pie chart distribusi cluster)
- Tabel seluruh provinsi beserta cluster
- Scatter plot interaktif
- Fitur pencarian provinsi
- Statistik cluster (mean values per feature)
- Form prediksi cluster manual untuk input data baru

## Fitur Utama Aplikasi

### Auto-Generate Dataset & Model
- `generate_cluster_data()` - memproses dataset mentah BPS, merge, scaling, train K-Means, dan save hasil
- `ensure_data()` - mengecek apakah output sudah ada, jika belum auto-generate

### ML Functions (tersedia di app.py)
- `parse_unemployment()` - parsing data pengangguran dari CSV BPS
- `parse_school()` - parsing data lama sekolah dari CSV BPS
- `parse_poverty()` - parsing data kemiskinan dari CSV BPS
- `build_features()` - merge semua dataset menjadi features untuk clustering
- `save_elbow_plot()` - generate visualisasi metode elbow untuk menentukan k optimal
- `perform_clustering()` - training K-Means dengan k=3 dan assign nama cluster
- `predict_cluster()` - prediksi cluster untuk input data baru menggunakan model yang sudah dilatih

### Dashboard Flask Routes
- `/` - dashboard utama dengan visualisasi dan tabel
- `/predict` - API endpoint untuk prediksi cluster
- `/elbow_plot` - menampilkan plot elbow method

## Catatan KNIME

Dokumentasi lengkap tentang workflow KNIME dan penjelasan setiap node tersedia di folder `knime/`.

## Interpretasi Singkat

- Cluster `Tinggi` menunjukkan provinsi dengan tingkat kemiskinan dan pengangguran relatif tinggi, serta rata-rata lama sekolah lebih rendah.
- Cluster `Sedang` berada pada posisi menengah untuk semua indikator.
- Cluster `Rendah` menunjukkan provinsi yang relatif lebih baik pada indikator sosial ekonomi (pengangguran lebih rendah, lama sekolah lebih tinggi).

Silakan buka `Docs/Laporan_Implementasi.md` untuk analisis lebih lengkap.
