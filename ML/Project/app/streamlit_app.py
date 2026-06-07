from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "output" / "cluster_results.csv"
MODEL_DIR = ROOT / "output"

st.set_page_config(
    page_title="Dashboard Segmentasi Kemiskinan Indonesia",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    if "Cluster_Name" not in df.columns:
        df["Cluster_Name"] = df["Cluster_Label"].astype(str)
    df["Cluster_Name"] = df["Cluster_Name"].astype(str)
    df["Provinsi"] = df["Provinsi"].astype(str)
    return df


@st.cache_resource
def load_model():
    try:
        scaler = joblib.load(MODEL_DIR / "scaler.pkl")
        kmeans = joblib.load(MODEL_DIR / "kmeans_model.pkl")
        cluster_map = joblib.load(MODEL_DIR / "cluster_name_map.pkl")
        return scaler, kmeans, cluster_map
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None


def make_excel_download(df: pd.DataFrame):
    from io import BytesIO
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Hasil Klaster")
    return output.getvalue()


def predict_cluster(unemployment, school, poverty, scaler, kmeans, cluster_map):
    input_data = np.array([[unemployment, school, poverty]])
    input_scaled = scaler.transform(input_data)
    cluster_label = kmeans.predict(input_scaled)[0]
    cluster_name = cluster_map[cluster_label]
    return cluster_label, cluster_name


def main():
    st.title("Dashboard Klustering Tingkat Kemiskinan Provinsi Indonesia")
    st.markdown("---")
    
    df = load_data()
    scaler, kmeans, cluster_map = load_model()

    total_provinsi = len(df)
    cluster_counts = df["Cluster_Name"].value_counts().reindex(["Rendah", "Sedang", "Tinggi"]).fillna(0).astype(int)

    st.subheader("Ringkasan Utama")
    cols = st.columns(4)
    cols[0].metric("Jumlah Provinsi", total_provinsi)
    cols[1].metric("Cluster Rendah", int(cluster_counts.get("Rendah", 0)))
    cols[2].metric("Cluster Sedang", int(cluster_counts.get("Sedang", 0)))
    cols[3].metric("Cluster Tinggi", int(cluster_counts.get("Tinggi", 0)))

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Dashboard",
        "Pencarian & Filter",
        "Data Lengkap",
        "Prediksi Manual",
        "Export"
    ])

    with tab1:
        st.subheader("Visualisasi Hasil Clustering")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Distribusi Cluster")
            fig_pie = px.pie(
                df,
                names="Cluster_Name",
                title="Persentase Provinsi per Cluster",
                hole=0.4,
                color_discrete_map={"Rendah": "#2ecc71", "Sedang": "#f39c12", "Tinggi": "#e74c3c"},
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.write("Statistik Rata-rata per Cluster")
            avg_values = df.groupby("Cluster_Name")[
                ["Tingkat_Pengangguran", "Rata_Rata_Lama_Sekolah", "Garis_Kemiskinan"]
            ].mean().round(2)
            st.dataframe(avg_values, use_container_width=True)

        st.markdown("---")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.write("Scatter Plot: Pengangguran vs Garis Kemiskinan")
            fig_scatter = px.scatter(
                df,
                x="Tingkat_Pengangguran",
                y="Garis_Kemiskinan",
                color="Cluster_Name",
                hover_data=["Provinsi", "Rata_Rata_Lama_Sekolah"],
                title="Hubungan Pengangguran dan Garis Kemiskinan",
                color_discrete_map={"Rendah": "#2ecc71", "Sedang": "#f39c12", "Tinggi": "#e74c3c"},
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col4:
            st.write("Histogram: Rata-rata Lama Sekolah")
            fig_hist = px.histogram(
                df,
                x="Rata_Rata_Lama_Sekolah",
                color="Cluster_Name",
                nbins=12,
                title="Distribusi Lama Sekolah per Cluster",
                color_discrete_map={"Rendah": "#2ecc71", "Sedang": "#f39c12", "Tinggi": "#e74c3c"},
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    with tab2:
        st.subheader("Cari dan Filter Provinsi")
        
        col_search, col_filter = st.columns(2)
        
        with col_search:
            st.write("Pencarian Provinsi")
            provinsi_search = st.text_input("Ketik nama provinsi:", value="", placeholder="Contoh: JAWA, BALI, ACEH")
            if provinsi_search:
                matches = df[df["Provinsi"].str.contains(provinsi_search.upper(), case=False, na=False)]
                st.write(f"Ditemukan {len(matches)} provinsi")
                st.dataframe(matches[["Provinsi", "Tingkat_Pengangguran", "Rata_Rata_Lama_Sekolah", "Garis_Kemiskinan", "Cluster_Name"]], use_container_width=True)
            else:
                st.info("Ketik nama provinsi untuk mencari")
        
        with col_filter:
            st.write("Filter Cluster")
            selected_clusters = st.multiselect(
                "Pilih cluster yang ingin ditampilkan:",
                options=["Rendah", "Sedang", "Tinggi"],
                default=["Rendah", "Sedang", "Tinggi"],
            )
            
            if selected_clusters:
                filtered_df = df[df["Cluster_Name"].isin(selected_clusters)]
                st.write(f"Menampilkan {len(filtered_df)} provinsi")
                
                st.dataframe(
                    filtered_df[["Provinsi", "Tingkat_Pengangguran", "Rata_Rata_Lama_Sekolah", "Garis_Kemiskinan", "Cluster_Name"]],
                    use_container_width=True
                )
                
                st.write("Statistik Cluster yang Dipilih:")
                stats = filtered_df.groupby("Cluster_Name")[
                    ["Tingkat_Pengangguran", "Rata_Rata_Lama_Sekolah", "Garis_Kemiskinan"]
                ].agg(["min", "max", "mean"]).round(2)
                st.dataframe(stats, use_container_width=True)

    with tab3:
        st.subheader("Data Lengkap Semua Provinsi")
        st.dataframe(df[["Provinsi", "Tingkat_Pengangguran", "Rata_Rata_Lama_Sekolah", "Garis_Kemiskinan", "Cluster_Label", "Cluster_Name"]], use_container_width=True)

    with tab4:
        st.subheader("Prediksi Cluster Perorangan")
        st.write("Masukkan nilai indikator untuk memprediksi cluster kategori mana yang sesuai.")
        
        col_input1, col_input2, col_input3 = st.columns(3)
        
        with col_input1:
            unemployment_input = st.slider(
                "Tingkat Pengangguran (%)",
                min_value=1.0,
                max_value=8.0,
                value=5.0,
                step=0.1
            )
        
        with col_input2:
            school_input = st.slider(
                "Rata-rata Lama Sekolah (tahun)",
                min_value=4.0,
                max_value=12.0,
                value=9.5,
                step=0.1
            )
        
        with col_input3:
            poverty_input = st.number_input(
                "Garis Kemiskinan (Rp/kapita/bulan)",
                min_value=400000,
                max_value=1500000,
                value=700000,
                step=10000
            )
        
        if st.button("Prediksi Cluster", use_container_width=True):
            if scaler and kmeans and cluster_map:
                cluster_label, cluster_name = predict_cluster(
                    unemployment_input, school_input, poverty_input, scaler, kmeans, cluster_map
                )
                
                st.success(f"Kategori Cluster: {cluster_name}")
                
                col_result1, col_result2 = st.columns(2)
                
                with col_result1:
                    st.write("Nilai Input:")
                    st.info(f"""
                    Tingkat Pengangguran: {unemployment_input}%
                    Rata-rata Lama Sekolah: {school_input} tahun
                    Garis Kemiskinan: Rp {poverty_input:,.0f}
                    """)
                
                with col_result2:
                    similar_provinces = df[df["Cluster_Label"] == cluster_label]
                    st.write("Provinsi yang sekelompok:")
                    st.write(f"Total {len(similar_provinces)} provinsi")
                    st.dataframe(
                        similar_provinces[["Provinsi", "Cluster_Name"]].head(10),
                        use_container_width=True,
                        hide_index=True
                    )
                
                st.markdown("---")
                st.write(f"Daftar lengkap provinsi Cluster {cluster_name}:")
                st.dataframe(
                    similar_provinces[["Provinsi", "Tingkat_Pengangguran", "Rata_Rata_Lama_Sekolah", "Garis_Kemiskinan"]],
                    use_container_width=True,
                    hide_index=True
                )
                
                st.markdown("---")
                st.write(f"Statistik Cluster {cluster_name}:")
                cluster_stats = similar_provinces[["Tingkat_Pengangguran", "Rata_Rata_Lama_Sekolah", "Garis_Kemiskinan"]].describe().round(2)
                st.dataframe(cluster_stats, use_container_width=True)
            else:
                st.error("Model belum tersedia. Jalankan generate_cluster_data.py terlebih dahulu.")

    with tab5:
        st.subheader("Ekspor Data")
        
        col_export1, col_export2, col_export3 = st.columns(3)
        
        with col_export1:
            st.write("Export Semua Data ke Excel")
            excel_data = make_excel_download(df)
            st.download_button(
                label="Download Excel",
                data=excel_data,
                file_name="hasil_clustering_provinsi.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        with col_export2:
            st.write("Export Semua Data ke CSV")
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="hasil_clustering_provinsi.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_export3:
            st.write("Export Ringkasan Cluster ke CSV")
            summary = df.groupby("Cluster_Name")[
                ["Tingkat_Pengangguran", "Rata_Rata_Lama_Sekolah", "Garis_Kemiskinan"]
            ].mean().round(2).reset_index()
            summary_csv = summary.to_csv(index=False)
            st.download_button(
                label="Download Ringkasan",
                data=summary_csv,
                file_name="ringkasan_cluster.csv",
                mime="text/csv",
                use_container_width=True
            )


if __name__ == "__main__":
    main()
