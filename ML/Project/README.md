# Implementasi K-Means Clustering untuk Segmentasi Tingkat Kemiskinan Provinsi di Indonesia

Proyek ini mengimplementasikan klastering K-Means untuk segmentasi tingkat kemiskinan provinsi Indonesia berdasarkan indikator sosial ekonomi. Proyek menggunakan KNIME sebagai tool data mining utama, dan Python Streamlit untuk dashboard interaktif.

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
├── app/                  # Aplikasi Streamlit dan script pembuat dataset
├── output/               # Hasil klastering dan visualisasi Elbow
├── laporan/              # Dokumentasi analisis dan interpretasi hasil
└── README.md

## Isi Folder

- `data/` : berisi dataset mentah dari `Docs/dataset` dan file `indonesia_sosial_ekonomi_2025.csv` hasil pemrosesan.
- `knime/` : berisi panduan workflow KNIME, node, pengaturan, dan visualisasi.
- `app/` : berisi aplikasi Streamlit `streamlit_app.py` dan script `generate_cluster_data.py` untuk menyiapkan dataset dan hasil klaster.
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

## Menjalankan Pra-pemrosesan dan Klastering

Jalankan script berikut untuk menggabungkan dataset, membuat fitur, dan menghasilkan hasil klastering:

```powershell
python app\generate_cluster_data.py
```

Output yang dihasilkan:

- `output\cluster_results.csv`
- `output\cluster_summary.csv`
- `output\elbow_plot.png`
- `data\indonesia_sosial_ekonomi_2025.csv`

## Menjalankan Dashboard Streamlit

Pastikan `output\cluster_results.csv` sudah tersedia. Lalu jalankan:

```powershell
streamlit run app\streamlit_app.py
```

Aplikasi dashboard akan tersedia di browser, menampilkan ringkasan klaster, tabel provinsi, distribusi cluster, scatter plot, pencarian provinsi, statistik cluster, dan fitur ekspor ke Excel.

## Catatan KNIME

Dokumentasi lengkap tentang workflow KNIME dan penjelasan setiap node tersedia di folder `knime/`.

## Interpretasi Singkat

- Cluster `Tinggi` menunjukkan provinsi dengan tingkat kemiskinan dan pengangguran relatif tinggi, IPM rendah, dan pengeluaran per kapita lebih kecil.
- Cluster `Sedang` berada pada posisi menengah.
- Cluster `Rendah` menunjukkan provinsi yang relatif lebih baik pada indikator sosial ekonomi.

Silakan buka `laporan/Laporan_Implementasi.md` untuk analisis lebih lengkap.
