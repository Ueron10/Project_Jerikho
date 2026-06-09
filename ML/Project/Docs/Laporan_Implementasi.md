# Laporan Implementasi K-Means Clustering

## Latar Belakang

Proyek ini mengimplementasikan K-Means clustering untuk segmentasi provinsi di Indonesia berdasarkan indikator sosial ekonomi. Tujuannya adalah mengelompokkan provinsi ke dalam cluster `Rendah`, `Sedang`, dan `Tinggi` terkait risiko kemiskinan dan kualitas hidup.

## Dataset

Dataset awal disusun dari tiga sumber:
- Tingkat Pengangguran Terbuka per provinsi
- Rata-rata Lama Sekolah per provinsi
- Garis Kemiskinan per provinsi (per kapita), untuk data perkotaan dan perdesaan

Analisis hanya menggunakan tiga indikator utama tersebut untuk klastering: `Tingkat_Pengangguran`, `Rata_Rata_Lama_Sekolah`, dan `Garis_Kemiskinan`.

## Preprocessing Data

Tahapan preprocessing yang dilakukan:
- Menggabungkan dataset berdasarkan `Provinsi`
- Menangani missing value dengan nilai rata-rata
- Menormalkan fitur numerik sebelum klastering
- Menghitung fitur baru:
  - `Persentase_Penduduk_Miskin`
  - `IPM`
  - `Pengeluaran_PerKapita`

## Penentuan Jumlah Cluster (Metode Elbow)

Metode Elbow digunakan untuk menentukan jumlah cluster terbaik. Model K-Means diuji pada k = 1 hingga k = 7, dan nilai inersia dicatat. Titik balik pada grafik Elbow menunjukkan bahwa `k = 3` adalah pilihan yang paling sesuai.

## Hasil Klastering

Hasil klastering dibagi menjadi tiga kelompok:
- `Tinggi`: provinsi dengan `Persentase_Penduduk_Miskin` tinggi, `Tingkat_Pengangguran` tinggi, `IPM` rendah, dan `Pengeluaran_PerKapita` relatif rendah.
- `Sedang`: provinsi dengan kondisi menengah pada semua indikator.
- `Rendah`: provinsi dengan tingkat kemiskinan dan pengangguran lebih rendah, IPM lebih tinggi, dan pengeluaran per kapita relatif lebih besar.

## Interpretasi Cluster

- Cluster `Tinggi` memerlukan perhatian prioritas dalam kebijakan pengentasan kemiskinan, program pendidikan, dan penciptaan lapangan kerja.
- Cluster `Sedang` adalah kelompok transisi; negara bagian ini dapat ditingkatkan melalui intervensi sosial yang ditargetkan.
- Cluster `Rendah` menunjukkan provinsi yang relatif lebih stabil dari segi sosial ekonomi, tetapi masih bisa dijaga agar tidak turun ke cluster lain.

## Visualisasi

Dashboard Streamlit menampilkan:
- Ringkasan jumlah provinsi per cluster
- Grafik distribusi cluster
- Scatter plot `Pengeluaran Per Kapita` vs `IPM`
- Tabel seluruh provinsi beserta cluster
- Fitur pencarian provinsi
- Statistik cluster

## Kesimpulan

Implementasi ini menunjukkan bahwa K-Means dapat digunakan untuk mengelompokkan provinsi berdasarkan indikator sosial ekonomi meskipun beberapa variabel perlu diestimasi. Hasil clustering dapat membantu perancang kebijakan fokus pada provinsi dengan kebutuhan sosial ekonomi paling tinggi.
