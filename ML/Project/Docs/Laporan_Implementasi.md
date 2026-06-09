# Laporan Implementasi K-Means Clustering

## Latar Belakang

Proyek ini mengimplementasikan K-Means clustering untuk segmentasi provinsi di Indonesia berdasarkan indikator sosial ekonomi. Tujuannya adalah mengelompokkan provinsi ke dalam cluster `Rendah`, `Sedang`, dan `Tinggi` terkait risiko kemiskinan dan kualitas hidup berdasarkan tiga indikator utama dari BPS.

## Dataset

Dataset awal disusun dari tiga sumber data BPS tahun 2025:
- Tingkat Pengangguran Terbuka per provinsi (data Agustus 2025)
- Rata-rata Lama Sekolah per provinsi (penduduk usia 15 tahun ke atas)
- Garis Kemiskinan per provinsi per kapita per bulan (rata-rata Semester 2 Perkotaan dan Perdesaan)

Analisis hanya menggunakan tiga indikator utama tersebut untuk klastering:
- `Tingkat_Pengangguran` - persentase pengangguran terbuka
- `Rata_Rata_Lama_Sekolah` - rata-rata tahun sekolah
- `Garis_Kemiskinan` - garis kemiskinan dalam Rupiah per kapita per bulan

## Preprocessing Data

Tahapan preprocessing yang dilakukan:

1. **Parsing Dataset Mentah**
   - `parse_unemployment()`: Membaca file CSV BPS pengangguran, skip header rows, mengambil kolom Provinsi dan Agustus, membersihkan whitespace, convert ke numeric
   - `parse_school()`: Membaca file CSV BPS lama sekolah, skip header rows, mengambil kolom Provinsi dan Rata-Rata Lama Sekolah, membersihkan whitespace, convert ke numeric
   - `parse_poverty()`: Membaca file CSV BPS kemiskinan, skip header rows, mengambil 7 kolom (Provinsi, Perkotaan_Sem1, Perkotaan_Sem2, Tahunan_Perkotaan, Perdesaan_Sem1, Perdesaan_Sem2, Tahunan_Perdesaan), menangani nilai "-", menghitung rata-rata Semester 2 perkotaan dan perdesaan sebagai Garis Kemiskinan

2. **Merge Dataset**
   - `build_features()`: Menggabungkan ketiga dataset berdasarkan kolom `Provinsi` menggunakan inner join untuk memastikan hanya provinsi yang lengkap datanya yang diproses

3. **Feature Scaling**
   - StandardScaler digunakan untuk menormalkan fitur numerik sebelum klastering agar setiap fitur memiliki skala yang seimbang

Tidak ada handling missing value dengan rata-rata karena data BPS yang digunakan sudah lengkap untuk provinsi-provinsi yang ada di semua dataset.

## Penentuan Jumlah Cluster (Metode Elbow)

Metode Elbow digunakan untuk menentukan jumlah cluster terbaik. Model K-Means diuji pada k = 1 hingga k = 7, dan nilai inersia dicerti untuk setiap k. Fungsi `save_elbow_plot()` menghasilkan visualisasi grafik elbow yang menunjukkan titik balik pada `k = 3` sebagai pilihan yang paling sesuai.

## Implementasi K-Means

Fungsi `perform_clustering()` menjalankan proses klastering:
1. StandardScaler untuk scaling fitur (kecuali kolom Provinsi)
2. KMeans dengan n_clusters=3, random_state=42, n_init="auto"
3. Prediksi cluster labels untuk semua data
4. Mapping numeric cluster labels ke nama cluster (Rendah, Sedang, Tinggi) berdasarkan severity rata-rata Tingkat_Pengangguran dan Garis_Kemiskinan
5. Menyimpan hasil ke CSV dan model ke file pickle

## Hasil Klastering

Hasil klastering dibagi menjadi tiga kelompok:
- `Tinggi`: provinsi dengan kombinasi Tingkat Pengangguran dan Garis Kemiskinan yang relatif tinggi
- `Sedang`: provinsi dengan kondisi menengah pada semua indikator
- `Rendah`: provinsi dengan tingkat kemiskinan dan pengangguran relatif lebih rendah

## Interpretasi Cluster

- Cluster `Tinggi` memerlukan perhatian prioritas dalam kebijakan pengentasan kemiskinan, program pendidikan, dan penciptaan lapangan kerja
- Cluster `Sedang` adalah kelompok transisi; provinsi ini dapat ditingkatkan melalui intervensi sosial yang ditargetkan
- Cluster `Rendah` menunjukkan provinsi yang relatif lebih stabil dari segi sosial ekonomi, tetapi masih bisa dijaga agar tidak turun ke cluster lain

## Implementasi Teknis

### Flask Web Application
Seluruh implementasi ML terdapat dalam satu file `app/app.py`:
- Auto-generate dataset dan model saat pertama kali dijalankan
- Dashboard web dengan Flask framework
- Visualisasi interaktif menggunakan Plotly
- Prediksi cluster untuk input data baru

### Fungsi-fungsi Utama
- `generate_cluster_data()`: Orchestrate seluruh proses dari parsing dataset hingga saving model
- `ensure_data()`: Check apakah output sudah ada, jika belum auto-generate
- `load_data()`: Load hasil klastering dari CSV dengan caching
- `load_model()`: Load model scaler, kmeans, dan cluster mapping dengan caching
- `predict_cluster()`: Prediksi cluster untuk input baru menggunakan model yang sudah dilatih

## Visualisasi

Dashboard Flask menampilkan:
- Pie chart distribusi cluster (persentase provinsi per cluster)
- Scatter plot interaktif Tingkat Pengangguran vs Rata-Rata Lama Sekolah dengan color coding per cluster
- Tabel seluruh provinsi beserta cluster dan nilai fitur
- Fitur pencarian provinsi untuk filter tabel
- Statistik ringkas per cluster (mean values untuk setiap fitur)
- Form prediksi cluster manual untuk input data baru

## Kesimpulan

Implementasi ini menunjukkan bahwa K-Means dapat digunakan untuk mengelompokkan provinsi berdasarkan indikator sosial ekonomi dari data BPS. Hasil clustering dapat membantu perancang kebijakan fokus pada provinsi dengan kebutuhan sosial ekonomi paling tinggi. Penggunaan Flask sebagai web framework memudahkan akses dan interaksi dengan hasil analisis secara real-time.
