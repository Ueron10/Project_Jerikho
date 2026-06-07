from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

UNEMPLOYMENT_FILE = DATA_DIR / "Tingkat Pengangguran Terbuka Menurut Provinsi, 2025.csv"
SCHOOL_FILE = DATA_DIR / "Rata-Rata Lama Sekolah Penduduk Umur 15 Tahun ke Atas Menurut Provinsi, 2025.csv"
POVERTY_FILE = DATA_DIR / "Garis Kemiskinan (Rupiah_Kapita_Bulan) Menurut Provinsi dan Daerah , 2025.csv"


def parse_unemployment(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, skiprows=4)
    df = df.iloc[:, :4]
    df.columns = ["Provinsi", "Februari", "Agustus", "Tahunan"]
    df = df.dropna(subset=["Provinsi"])
    df["Provinsi"] = df["Provinsi"].astype(str).str.strip()
    df["Agustus"] = pd.to_numeric(df["Agustus"], errors="coerce")
    return df[["Provinsi", "Agustus"]].rename(columns={"Agustus": "Tingkat_Pengangguran"})


def parse_school(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, skiprows=2)
    df = df.iloc[:, :2]
    df.columns = ["Provinsi", "Rata_Rata_Lama_Sekolah"]
    df["Provinsi"] = df["Provinsi"].astype(str).str.strip()
    df["Rata_Rata_Lama_Sekolah"] = pd.to_numeric(df["Rata_Rata_Lama_Sekolah"], errors="coerce")
    return df


def parse_poverty(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, skiprows=4)
    df = df.iloc[:, :7]
    df.columns = [
        "Provinsi",
        "Perkotaan_Sem1",
        "Perkotaan_Sem2",
        "Tahunan_Perkotaan",
        "Perdesaan_Sem1",
        "Perdesaan_Sem2",
        "Tahunan_Perdesaan",
    ]
    df["Provinsi"] = df["Provinsi"].astype(str).str.strip()

    def safe_float(value):
        try:
            return float(str(value).replace("-", ""))
        except Exception:
            return np.nan

    for col in [
        "Perkotaan_Sem1",
        "Perkotaan_Sem2",
        "Tahunan_Perkotaan",
        "Perdesaan_Sem1",
        "Perdesaan_Sem2",
        "Tahunan_Perdesaan",
    ]:
        df[col] = df[col].apply(safe_float)

    df["Garis_Kemiskinan"] = df[["Perkotaan_Sem2", "Perdesaan_Sem2"]].mean(axis=1)
    return df[["Provinsi", "Garis_Kemiskinan"]]


def build_features(unemployment: pd.DataFrame, school: pd.DataFrame, poverty: pd.DataFrame) -> pd.DataFrame:
    df = unemployment.merge(school, on="Provinsi", how="inner")
    df = df.merge(poverty, on="Provinsi", how="inner")

    return df[
        [
            "Provinsi",
            "Tingkat_Pengangguran",
            "Rata_Rata_Lama_Sekolah",
            "Garis_Kemiskinan",
        ]
    ]


def save_elbow_plot(features: pd.DataFrame, path: Path):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features.drop(columns=["Provinsi"]))
    inertias = []
    ks = list(range(1, 8))
    for k in ks:
        model = KMeans(n_clusters=k, random_state=42, n_init="auto")
        model.fit(X_scaled)
        inertias.append(model.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(ks, inertias, marker="o", linestyle="-", color="#2c3e50")
    plt.title("Metode Elbow untuk Menentukan Jumlah Cluster Terbaik")
    plt.xlabel("Jumlah Cluster (k)")
    plt.ylabel("Inersia")
    plt.xticks(ks)
    plt.grid(True, alpha=0.4)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def perform_clustering(features: pd.DataFrame, path_csv: Path, path_summary: Path):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features.drop(columns=["Provinsi"]))
    kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X_scaled)

    result = features.copy()
    result["Cluster_Label"] = labels
    summary = (
        result.groupby("Cluster_Label")[
            [
                "Tingkat_Pengangguran",
                "Rata_Rata_Lama_Sekolah",
                "Garis_Kemiskinan",
            ]
        ]
        .mean()
        .round(2)
    )

    severity = summary[["Tingkat_Pengangguran", "Garis_Kemiskinan"]].mean(axis=1)
    label_order = severity.sort_values(ascending=False).index.tolist()
    cluster_name_map = {
        label_order[0]: "Tinggi",
        label_order[1]: "Sedang",
        label_order[2]: "Rendah",
    }
    result["Cluster_Name"] = result["Cluster_Label"].map(cluster_name_map)
    result.to_csv(path_csv, index=False)
    summary.to_csv(path_summary)
    
    # Save scaler and kmeans model
    joblib.dump(scaler, OUTPUT_DIR / "scaler.pkl")
    joblib.dump(kmeans, OUTPUT_DIR / "kmeans_model.pkl")
    joblib.dump(cluster_name_map, OUTPUT_DIR / "cluster_name_map.pkl")
    
    return result, summary


def main():
    unemployment = parse_unemployment(UNEMPLOYMENT_FILE)
    school = parse_school(SCHOOL_FILE)
    poverty = parse_poverty(POVERTY_FILE)
    features = build_features(unemployment, school, poverty)

    data_path = DATA_DIR / "indonesia_sosial_ekonomi_2025.csv"
    features.to_csv(data_path, index=False)

    save_elbow_plot(features, OUTPUT_DIR / "elbow_plot.png")
    result, summary = perform_clustering(
        features,
        OUTPUT_DIR / "cluster_results.csv",
        OUTPUT_DIR / "cluster_summary.csv",
    )

    print("Data mentah berhasil dibuat:", data_path)
    print("Hasil elbow plot disimpan di:", OUTPUT_DIR / "elbow_plot.png")
    print("Hasil klastering tersimpan di:", OUTPUT_DIR / "cluster_results.csv")
    print("Ringkasan klaster tersimpan di:", OUTPUT_DIR / "cluster_summary.csv")


if __name__ == "__main__":
    main()
