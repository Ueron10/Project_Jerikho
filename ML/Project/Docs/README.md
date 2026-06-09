# Implementasi K-Means Clustering untuk Segmentasi Tingkat Kemiskinan Provinsi di Indonesia

Proyek ini mengimplementasikan klastering K-Means untuk segmentasi tingkat kemiskinan provinsi Indonesia berdasarkan indikator sosial ekonomi. Proyek menggunakan KNIME sebagai tool data mining utama, dan Python Flask untuk dashboard web interaktif.

Dataset yang digunakan mencakup tiga indikator asli provinsi berikut:
- Nama Provinsi
- Tingkat Pengangguran Terbuka (Agustus 2025)
- Rata-rata Lama Sekolah (2025)
- Garis Kemiskinan Per Kapita (rata-rata Semester 2 Perkotaan dan Perdesaan)

Klastering dan visualisasi hanya menggunakan ketiga fitur utama tersebut.

## Struktur Folder

Project/
├── data/                 # Dataset mentah dan file hasil preprocessing
├── knime/                # Deskripsi workflow KNIME dan dokumentasi node
├── app/                  # Aplikasi web Flask dan script pembuat dataset
├── output/               # Hasil klastering dan visualisasi Elbow
├── laporan/              # Dokumentasi analisis dan interpretasi hasil
└── README.md

## Isi Folder

- `data/` : berisi dataset mentah dari `Docs/dataset` dan file `indonesia_sosial_ekonomi_2025.csv` hasil pemrosesan.
- `knime/` : berisi panduan workflow KNIME, node, pengaturan, dan visualisasi.
- `app/` : berisi aplikasi web Flask `app.py` (beserta `templates/` dan `static/`). `app.py` sekaligus menyiapkan dataset, model, dan hasil klaster secara otomatis saat pertama dijalankan.
- `Docs/dataset/` : berisi dataset mentah BPS (pengangguran, lama sekolah, garis kemiskinan) yang dipakai untuk membangun hasil klastering.
- `output/` : berisi file `cluster_results.csv`, `cluster_summary.csv`, dan `elbow_plot.png`.
- `laporan/` : berisi laporan analisis dan interpretasi.

## Instalasi

1. Buka terminal di folder `Project`.
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
python app\app.py
```

Saat pertama dijalankan, `app.py` otomatis membaca dataset mentah dari `Docs/dataset/`, membangun fitur, melatih K-Means, dan menyimpan output berikut bila belum ada:

- `output\cluster_results.csv`
- `output\cluster_summary.csv`
- `output\elbow_plot.png`
- `output\scaler.pkl`, `output\kmeans_model.pkl`, `output\cluster_name_map.pkl`
- `data\indonesia_sosial_ekonomi_2025.csv`

Kalau ingin membangun ulang dari awal, hapus isi folder `output/` lalu jalankan lagi.

Buka `http://localhost:5000` di browser. Aplikasi dashboard menampilkan ringkasan klaster, tabel provinsi, distribusi cluster, scatter plot, pencarian provinsi, statistik cluster, dan prediksi cluster manual.

## Catatan KNIME

Dokumentasi lengkap tentang workflow KNIME dan penjelasan setiap node tersedia di folder `knime/`.

## Interpretasi Singkat

- Cluster `Tinggi` menunjukkan provinsi dengan tingkat kemiskinan dan pengangguran relatif tinggi, IPM rendah, dan pengeluaran per kapita lebih kecil.
- Cluster `Sedang` berada pada posisi menengah.
- Cluster `Rendah` menunjukkan provinsi yang relatif lebih baik pada indikator sosial ekonomi.

Silakan buka `laporan/Laporan_Implementasi.md` untuk analisis lebih lengkap.
