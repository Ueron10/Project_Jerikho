from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
from flask import Flask, jsonify, render_template, request
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
DATA_FILE = OUTPUT_DIR / "cluster_results.csv"
MODEL_DIR = OUTPUT_DIR

# Raw datasets shipped with the repo (used to build the clustering output on first run).
DOCS_DATASET_DIR = ROOT / "Docs" / "dataset"
UNEMPLOYMENT_FILE = "Tingkat Pengangguran Terbuka Menurut Provinsi, 2025.csv"
SCHOOL_FILE = "Rata-Rata Lama Sekolah Penduduk Umur 15 Tahun ke Atas Menurut Provinsi, 2025.csv"
POVERTY_FILE = "Garis Kemiskinan (Rupiah_Kapita_Bulan) Menurut Provinsi dan Daerah , 2025.csv"

CLUSTER_ORDER = ["Rendah", "Sedang", "Tinggi"]
CLUSTER_COLORS = {"Rendah": "#2ecc71", "Sedang": "#f39c12", "Tinggi": "#e74c3c"}

app = Flask(__name__, template_folder="templates", static_folder="static")


# ---------------------------------------------------------------------------
# Data preparation (clustering) -- runs automatically when outputs are missing.
# ---------------------------------------------------------------------------
def _find_dataset(filename: str) -> Path:
    candidate = DATA_DIR / filename
    if candidate.exists():
        return candidate
    return DOCS_DATASET_DIR / filename


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
    df["Rata_Rata_Lama_Sekolah"] = pd.to_numeric(
        df["Rata_Rata_Lama_Sekolah"], errors="coerce"
    )
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


def build_features() -> pd.DataFrame:
    unemployment = parse_unemployment(_find_dataset(UNEMPLOYMENT_FILE))
    school = parse_school(_find_dataset(SCHOOL_FILE))
    poverty = parse_poverty(_find_dataset(POVERTY_FILE))

    df = unemployment.merge(school, on="Provinsi", how="inner")
    df = df.merge(poverty, on="Provinsi", how="inner")
    return df[
        ["Provinsi", "Tingkat_Pengangguran", "Rata_Rata_Lama_Sekolah", "Garis_Kemiskinan"]
    ]


def save_elbow_plot(features: pd.DataFrame, path: Path):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features.drop(columns=["Provinsi"]))
    ks = list(range(1, 8))
    inertias = []
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


def perform_clustering(features: pd.DataFrame):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features.drop(columns=["Provinsi"]))
    kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X_scaled)

    result = features.copy()
    result["Cluster_Label"] = labels
    summary = (
        result.groupby("Cluster_Label")[
            ["Tingkat_Pengangguran", "Rata_Rata_Lama_Sekolah", "Garis_Kemiskinan"]
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

    result.to_csv(DATA_FILE, index=False)
    summary.to_csv(OUTPUT_DIR / "cluster_summary.csv")
    joblib.dump(scaler, OUTPUT_DIR / "scaler.pkl")
    joblib.dump(kmeans, OUTPUT_DIR / "kmeans_model.pkl")
    joblib.dump(cluster_name_map, OUTPUT_DIR / "cluster_name_map.pkl")
    return result


def generate_cluster_data():
    """Build features, run K-Means, and persist datasets/models/plot."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    features = build_features()
    features.to_csv(DATA_DIR / "indonesia_sosial_ekonomi_2025.csv", index=False)
    save_elbow_plot(features, OUTPUT_DIR / "elbow_plot.png")
    perform_clustering(features)
    print("Data klastering & model berhasil dibuat di", OUTPUT_DIR)


def ensure_data(force: bool = False):
    """Generate clustering outputs/models if they are missing (or if forced)."""
    required = [
        DATA_FILE,
        OUTPUT_DIR / "scaler.pkl",
        OUTPUT_DIR / "kmeans_model.pkl",
        OUTPUT_DIR / "cluster_name_map.pkl",
    ]
    if force or not all(p.exists() for p in required):
        print("Menyiapkan data klastering & model...")
        generate_cluster_data()
    return all(p.exists() for p in required)


# In-memory caches so the dashboard and predictions don't re-read disk per request.
_DATA_CACHE = None
_MODEL_CACHE = None


def load_data():
    global _DATA_CACHE
    if _DATA_CACHE is None:
        df = pd.read_csv(DATA_FILE)
        if "Cluster_Name" not in df.columns:
            df["Cluster_Name"] = df["Cluster_Label"].astype(str)
        df["Cluster_Name"] = df["Cluster_Name"].astype(str)
        df["Provinsi"] = df["Provinsi"].astype(str)
        _DATA_CACHE = df
    return _DATA_CACHE.copy()


def load_model():
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        try:
            scaler = joblib.load(MODEL_DIR / "scaler.pkl")
            kmeans = joblib.load(MODEL_DIR / "kmeans_model.pkl")
            cluster_map = joblib.load(MODEL_DIR / "cluster_name_map.pkl")
            _MODEL_CACHE = (scaler, kmeans, cluster_map)
        except Exception as e:
            print(f"Error loading model: {e}")
            return None, None, None
    return _MODEL_CACHE


def predict_cluster(unemployment, school, poverty, scaler, kmeans, cluster_map):
    input_data = np.array([[unemployment, school, poverty]])
    input_scaled = scaler.transform(input_data)
    cluster_label = kmeans.predict(input_scaled)[0]
    cluster_name = cluster_map[cluster_label]
    return int(cluster_label), str(cluster_name)


def fig_to_json(fig):
    return plotly.io.to_json(fig)


def build_charts(df: pd.DataFrame):
    fig_pie = px.pie(
        df,
        names="Cluster_Name",
        title="Persentase Provinsi per Cluster",
        hole=0.4,
        color="Cluster_Name",
        color_discrete_map=CLUSTER_COLORS,
    )

    fig_scatter = px.scatter(
        df,
        x="Tingkat_Pengangguran",
        y="Garis_Kemiskinan",
        color="Cluster_Name",
        hover_data=["Provinsi", "Rata_Rata_Lama_Sekolah"],
        title="Hubungan Pengangguran dan Garis Kemiskinan",
        color_discrete_map=CLUSTER_COLORS,
    )

    fig_hist = px.histogram(
        df,
        x="Rata_Rata_Lama_Sekolah",
        color="Cluster_Name",
        nbins=12,
        title="Distribusi Lama Sekolah per Cluster",
        color_discrete_map=CLUSTER_COLORS,
    )

    return {
        "pie": fig_to_json(fig_pie),
        "scatter": fig_to_json(fig_scatter),
        "hist": fig_to_json(fig_hist),
    }


def records(df: pd.DataFrame):
    return df.to_dict(orient="records")


@app.route("/")
def index():
    df = load_data()
    total_provinsi = len(df)
    cluster_counts = (
        df["Cluster_Name"].value_counts().reindex(CLUSTER_ORDER).fillna(0).astype(int)
    )

    avg_values = (
        df.groupby("Cluster_Name")[
            ["Tingkat_Pengangguran", "Rata_Rata_Lama_Sekolah", "Garis_Kemiskinan"]
        ]
        .mean()
        .round(2)
        .reindex(CLUSTER_ORDER)
        .dropna(how="all")
        .reset_index()
    )

    charts = build_charts(df)

    full_cols = [
        "Provinsi",
        "Tingkat_Pengangguran",
        "Rata_Rata_Lama_Sekolah",
        "Garis_Kemiskinan",
        "Cluster_Label",
        "Cluster_Name",
    ]

    return render_template(
        "index.html",
        total_provinsi=total_provinsi,
        cluster_counts=cluster_counts.to_dict(),
        avg_values=records(avg_values),
        charts=charts,
        full_data=records(df[full_cols]),
    )


@app.route("/api/search")
def api_search():
    query = request.args.get("q", "").strip()
    df = load_data()
    cols = [
        "Provinsi",
        "Tingkat_Pengangguran",
        "Rata_Rata_Lama_Sekolah",
        "Garis_Kemiskinan",
        "Cluster_Name",
    ]
    if not query:
        return jsonify({"count": 0, "rows": []})
    matches = df[df["Provinsi"].str.contains(query.upper(), case=False, na=False)]
    return jsonify({"count": int(len(matches)), "rows": records(matches[cols])})


@app.route("/api/filter")
def api_filter():
    selected = request.args.getlist("clusters")
    if not selected:
        selected = CLUSTER_ORDER
    df = load_data()
    cols = [
        "Provinsi",
        "Tingkat_Pengangguran",
        "Rata_Rata_Lama_Sekolah",
        "Garis_Kemiskinan",
        "Cluster_Name",
    ]
    filtered = df[df["Cluster_Name"].isin(selected)]

    stats = (
        filtered.groupby("Cluster_Name")[
            ["Tingkat_Pengangguran", "Rata_Rata_Lama_Sekolah", "Garis_Kemiskinan"]
        ]
        .agg(["min", "max", "mean"])
        .round(2)
    )
    stats.columns = [f"{metric}_{agg}" for metric, agg in stats.columns]
    stats = stats.reset_index()

    return jsonify(
        {
            "count": int(len(filtered)),
            "rows": records(filtered[cols]),
            "stats": records(stats),
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    scaler, kmeans, cluster_map = load_model()
    if not (scaler is not None and kmeans is not None and cluster_map is not None):
        return (
            jsonify(
                {
                    "error": "Model belum tersedia. Jalankan generate_cluster_data.py terlebih dahulu."
                }
            ),
            400,
        )

    data = request.get_json(silent=True) or {}
    try:
        unemployment = float(data.get("unemployment"))
        school = float(data.get("school"))
        poverty = float(data.get("poverty"))
    except (TypeError, ValueError):
        return jsonify({"error": "Input tidak valid."}), 400

    cluster_label, cluster_name = predict_cluster(
        unemployment, school, poverty, scaler, kmeans, cluster_map
    )

    df = load_data()
    similar = df[df["Cluster_Label"] == cluster_label]

    cluster_stats = (
        similar[["Tingkat_Pengangguran", "Rata_Rata_Lama_Sekolah", "Garis_Kemiskinan"]]
        .describe()
        .round(2)
    )
    cluster_stats.insert(0, "Statistik", cluster_stats.index)

    return jsonify(
        {
            "cluster_label": cluster_label,
            "cluster_name": cluster_name,
            "input": {
                "unemployment": unemployment,
                "school": school,
                "poverty": poverty,
            },
            "count": int(len(similar)),
            "provinces": records(
                similar[
                    [
                        "Provinsi",
                        "Tingkat_Pengangguran",
                        "Rata_Rata_Lama_Sekolah",
                        "Garis_Kemiskinan",
                    ]
                ]
            ),
            "stats": records(cluster_stats),
        }
    )


ensure_data()


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
