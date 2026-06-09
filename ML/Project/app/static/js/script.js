const CLUSTER_COLS = {
    Provinsi: "Provinsi",
    Tingkat_Pengangguran: "Pengangguran",
    Rata_Rata_Lama_Sekolah: "Lama Sekolah",
    Garis_Kemiskinan: "Garis Kemiskinan",
    Cluster_Name: "Cluster",
};

function rupiah(value) {
    const num = Number(value);
    if (Number.isNaN(num)) return value;
    return "Rp " + num.toLocaleString("id-ID");
}

function buildTable(rows, columns) {
    if (!rows || rows.length === 0) {
        return '<p class="info">Tidak ada data.</p>';
    }
    let html = '<table class="data-table"><thead><tr>';
    columns.forEach((col) => {
        html += `<th>${col.label}</th>`;
    });
    html += "</tr></thead><tbody>";
    rows.forEach((row) => {
        html += "<tr>";
        columns.forEach((col) => {
            html += `<td>${row[col.key]}</td>`;
        });
        html += "</tr>";
    });
    html += "</tbody></table>";
    return html;
}

// ---- Tabs ----
document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
        document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
        document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
        btn.classList.add("active");
        document.getElementById("tab-" + btn.dataset.tab).classList.add("active");
        // Plotly charts need a resize when their container becomes visible.
        window.dispatchEvent(new Event("resize"));
    });
});

// ---- Charts ----
function renderChart(elementId, dataId) {
    const raw = document.getElementById(dataId).textContent;
    const fig = JSON.parse(raw);
    Plotly.newPlot(elementId, fig.data, fig.layout, { responsive: true });
}

renderChart("chart-pie", "chart-pie-data");
renderChart("chart-scatter", "chart-scatter-data");
renderChart("chart-hist", "chart-hist-data");

// ---- Search ----
const searchInput = document.getElementById("searchInput");
let searchTimer = null;
searchInput.addEventListener("input", () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(runSearch, 250);
});

async function runSearch() {
    const q = searchInput.value.trim();
    const container = document.getElementById("searchResult");
    if (!q) {
        container.innerHTML = '<p class="info">Ketik nama provinsi untuk mencari</p>';
        return;
    }
    const res = await fetch(`/api/search?q=${encodeURIComponent(q)}`);
    const data = await res.json();
    const cols = [
        { key: "Provinsi", label: "Provinsi" },
        { key: "Tingkat_Pengangguran", label: "Pengangguran" },
        { key: "Rata_Rata_Lama_Sekolah", label: "Lama Sekolah" },
        { key: "Garis_Kemiskinan", label: "Garis Kemiskinan" },
        { key: "Cluster_Name", label: "Cluster" },
    ];
    container.innerHTML =
        `<p class="muted">Ditemukan ${data.count} provinsi</p>` + buildTable(data.rows, cols);
}

// ---- Filter ----
document.querySelectorAll(".filter-cluster").forEach((cb) => {
    cb.addEventListener("change", runFilter);
});

async function runFilter() {
    const selected = Array.from(document.querySelectorAll(".filter-cluster:checked")).map(
        (cb) => cb.value
    );
    const container = document.getElementById("filterResult");
    const params = selected.map((c) => `clusters=${encodeURIComponent(c)}`).join("&");
    const res = await fetch(`/api/filter?${params}`);
    const data = await res.json();

    const cols = [
        { key: "Provinsi", label: "Provinsi" },
        { key: "Tingkat_Pengangguran", label: "Pengangguran" },
        { key: "Rata_Rata_Lama_Sekolah", label: "Lama Sekolah" },
        { key: "Garis_Kemiskinan", label: "Garis Kemiskinan" },
        { key: "Cluster_Name", label: "Cluster" },
    ];

    let html = `<p class="muted">Menampilkan ${data.count} provinsi</p>`;
    html += buildTable(data.rows, cols);

    if (data.stats && data.stats.length > 0) {
        const statCols = [{ key: "Cluster_Name", label: "Cluster" }];
        Object.keys(data.stats[0])
            .filter((k) => k !== "Cluster_Name")
            .forEach((k) => statCols.push({ key: k, label: k.replace(/_/g, " ") }));
        html += '<p class="muted">Statistik Cluster yang Dipilih:</p>';
        html += buildTable(data.stats, statCols);
    }
    container.innerHTML = html;
}

runFilter();

// ---- Predict ----
document.getElementById("predictBtn").addEventListener("click", runPredict);

async function runPredict() {
    const container = document.getElementById("predictResult");
    const payload = {
        unemployment: parseFloat(document.getElementById("unemployment").value),
        school: parseFloat(document.getElementById("school").value),
        poverty: parseFloat(document.getElementById("poverty").value),
    };
    container.innerHTML = '<p class="muted">Memproses...</p>';

    const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });
    const data = await res.json();

    if (!res.ok) {
        container.innerHTML = `<p class="info">${data.error || "Terjadi kesalahan."}</p>`;
        return;
    }

    let html = `<div class="success-banner">Kategori Cluster: ${data.cluster_name}</div>`;
    html += '<div class="grid-2">';
    html += '<div class="card"><h3>Nilai Input</h3>';
    html += `<p>Tingkat Pengangguran: ${data.input.unemployment}%</p>`;
    html += `<p>Rata-rata Lama Sekolah: ${data.input.school} tahun</p>`;
    html += `<p>Garis Kemiskinan: ${rupiah(data.input.poverty)}</p></div>`;

    const provCols = [
        { key: "Provinsi", label: "Provinsi" },
        { key: "Tingkat_Pengangguran", label: "Pengangguran" },
        { key: "Rata_Rata_Lama_Sekolah", label: "Lama Sekolah" },
        { key: "Garis_Kemiskinan", label: "Garis Kemiskinan" },
    ];
    html += `<div class="card"><h3>Provinsi sekelompok (${data.count})</h3>`;
    html += `<div class="table-wrap">${buildTable(data.provinces, provCols)}</div></div>`;
    html += "</div>";

    if (data.stats && data.stats.length > 0) {
        const statCols = [{ key: "Statistik", label: "Statistik" }];
        Object.keys(data.stats[0])
            .filter((k) => k !== "Statistik")
            .forEach((k) => statCols.push({ key: k, label: k.replace(/_/g, " ") }));
        html += `<div class="card"><h3>Statistik Cluster ${data.cluster_name}</h3>`;
        html += buildTable(data.stats, statCols) + "</div>";
    }

    container.innerHTML = html;
}
