# Workflow KNIME: Segmentasi Kemiskinan Provinsi Indonesia

Dokumentasi ini menjelaskan workflow KNIME yang digunakan untuk melakukan preprocessing data, menentukan jumlah cluster menggunakan metode Elbow, melakukan klastering K-Means, dan mengekspor hasil.

## Node dan Alur Kerja

1. **CSV Reader**
   - Baca file dataset:
     - `Tingkat Pengangguran Terbuka Menurut Provinsi, 2025.csv`
     - `Rata-Rata Lama Sekolah Penduduk Umur 15 Tahun ke Atas Menurut Provinsi, 2025.csv`
     - `Garis Kemiskinan (Rupiah_Kapita_Bulan) Menurut Provinsi dan Daerah , 2025.csv`
   - Set delimiter `,` dan pastikan encoding UTF-8.

2. **String Manipulation**
   - Bersihkan nama provinsi dengan `strip()` dan normalisasi teks agar join konsisten.

3. **Column Filter**
   - Pilih kolom yang relevan: `Provinsi`, `Agustus` (Tingkat Pengangguran), `Rata_Rata_Lama_Sekolah`, dan `Garis_Kemiskinan`.

4. **Poverty Line Aggregation**
   - Hitung `Garis_Kemiskinan` sebagai rata-rata dari `Perkotaan_Sem2` dan `Perdesaan_Sem2`, karena data persentase penduduk miskin tidak disediakan langsung.

5. **Missing Value**
   - Tangani nilai kosong dengan strategi `Mean` atau `Previous Value` sesuai jumlah missing value.

6. **Joiner (Join)**
   - Gabungkan dataset berdasarkan kolom `Provinsi` secara inner join.
   - Pastikan kolom join terdiri dari nama provinsi yang sama.

6. **Math Formula**
   - Tidak diperlukan fitur turunan tambahan. Fokus pada tiga indikator utama:
     - `Tingkat_Pengangguran`
     - `Rata_Rata_Lama_Sekolah`
     - `Garis_Kemiskinan`

7. **Normalizer**
   - Normalisasi fitur numerik:
     - `Tingkat_Pengangguran`
     - `Rata_Rata_Lama_Sekolah`
     - `Garis_Kemiskinan`
   - Gunakan `Z-Score` atau `Min-Max`.

8. **K-Means**
   - Gunakan node `K-Means` untuk eksperimen dengan k = 1..7.
   - Simpan nilai inersia untuk setiap k.
   - Buat grafik Elbow dari nilai inersia.

9. **Line Plot (or JavaScript Scatter Plot)**
   - Visualisasikan hasil metode Elbow.
   - Pilih jumlah cluster terbaik berdasarkan titik “siku” (elbow).

10. **K-Means (final)**
    - Jalankan K-Means akhir dengan jumlah cluster terbaik (`k = 3`).
    - Tambahkan kolom hasil klaster: `Cluster_Label` dan `Cluster_Name`.

11. **Color Manager / Scatter Plot**
    - Visualisasikan provinsi berdasarkan dua indikator utama, misalnya:
      - `Pengeluaran_PerKapita` vs `IPM`
      - `Persentase_Penduduk_Miskin` vs `Tingkat_Pengangguran`
    - Warna berdasarkan `Cluster_Name`.

12. **Excel Writer**
    - Ekspor hasil klastering ke file Excel atau CSV.
    - Pastikan file output disimpan di folder `output/`.

## Penjelasan setiap node

- **CSV Reader**: membaca file CSV mentah. Penting untuk menangani baris header tidak standar.
- **String Manipulation**: memastikan nama provinsi konsisten sebelum penggabungan.
- **Column Filter**: menghapus kolom yang tidak relevan untuk menghindari noise.
- **Missing Value**: menggantikan nilai kosong agar model tidak gagal.
- **Joiner**: menggabungkan data dari tiga sumber berbeda menjadi satu dataset lengkap.
- **Math Formula**: membuat variabel baru yang diperlukan untuk segmen sosial ekonomi.
- **Normalizer**: menormalkan skala fitur agar K-Means bekerja optimal.
- **K-Means**: melakukan pembentukan cluster berdasarkan fitur numerik.
- **Line Plot**: membantu memilih jumlah cluster dengan Elbow.
- **Excel Writer**: menyimpan hasil akhir ke file yang bisa dibaca aplikasi Streamlit.

## Output yang diharapkan

- `output/cluster_results.csv`
- `output/cluster_summary.csv`
- `output/elbow_plot.png`

## Catatan

Karena dataset awal tersedia dalam format berbeda, workflow KNIME harus melakukan beberapa tahapan preprocessing sebelum klastering. Bagian `Math Formula` digunakan untuk menambahkan indikator penting yang tidak langsung tersedia di dataset mentah.
