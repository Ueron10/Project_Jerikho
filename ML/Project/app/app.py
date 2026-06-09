from pathlib import Path
from io import BytesIO

import joblib
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
from flask import Flask, jsonify, render_template, request, send_file

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "output" / "cluster_results.csv"
MODEL_DIR = ROOT / "output"

CLUSTER_ORDER = ["Rendah", "Sedang", "Tinggi"]
CLUSTER_COLORS = {"Rendah": "#2ecc71", "Sedang": "#f39c12", "Tinggi": "#e74c3c"}

app = Flask(__name__, template_folder="templates", static_folder="static")


def load_data():
    df = pd.read_csv(DATA_FILE)
    if "Cluster_Name" not in df.columns:
        df["Cluster_Name"] = df["Cluster_Label"].astype(str)
    df["Cluster_Name"] = df["Cluster_Name"].astype(str)
    df["Provinsi"] = df["Provinsi"].astype(str)
    return df


def load_model():
    try:
        scaler = joblib.load(MODEL_DIR / "scaler.pkl")
        kmeans = joblib.load(MODEL_DIR / "kmeans_model.pkl")
        cluster_map = joblib.load(MODEL_DIR / "cluster_name_map.pkl")
        return scaler, kmeans, cluster_map
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None


def predict_cluster(unemployment, school, poverty, scaler, kmeans, cluster_map):
    input_data = np.array([[unemployment, school, poverty]])
    input_scaled = scaler.transform(input_data)
    cluster_label = kmeans.predict(input_scaled)[0]
    cluster_name = cluster_map[cluster_label]
    return int(cluster_label), str(cluster_name)


def make_excel_download(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Hasil Klaster")
    return output.getvalue()


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


@app.route("/export/excel")
def export_excel():
    df = load_data()
    data = make_excel_download(df)
    return send_file(
        BytesIO(data),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name="hasil_clustering_provinsi.xlsx",
    )


@app.route("/export/csv")
def export_csv():
    df = load_data()
    return send_file(
        BytesIO(df.to_csv(index=False).encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name="hasil_clustering_provinsi.csv",
    )


@app.route("/export/summary")
def export_summary():
    df = load_data()
    summary = (
        df.groupby("Cluster_Name")[
            ["Tingkat_Pengangguran", "Rata_Rata_Lama_Sekolah", "Garis_Kemiskinan"]
        ]
        .mean()
        .round(2)
        .reset_index()
    )
    return send_file(
        BytesIO(summary.to_csv(index=False).encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name="ringkasan_cluster.csv",
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
