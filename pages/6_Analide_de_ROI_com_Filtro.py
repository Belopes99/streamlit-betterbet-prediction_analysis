import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json

# =========================
# SETUP / BIGQUERY
# =========================
credentials_info = json.loads(st.secrets["gcp"]["key"])
credentials = service_account.Credentials.from_service_account_info(credentials_info)
project_id = credentials_info["project_id"]
client = bigquery.Client(credentials=credentials, project=project_id)

st.title("Análise de ROI")

QUERY = """
SELECT *
FROM `betterbet-467621.betterbet.predictions`
WHERE DATE(match_date) >= DATE('2025-10-28')
"""

@st.cache_data
def run_query(sql: str) -> pd.DataFrame:
    return client.query(sql).to_dataframe()

df = run_query(QUERY)
df["match_date"] = pd.to_datetime(df["match_date"]).dt.tz_localize(None)

# =========================
# STATE (SIDEBAR FILTERS)
# =========================
if "filters_leagues" not in st.session_state:
    st.session_state["filters_leagues"] = ["Todas"]
if "filters_seasons" not in st.session_state:
    st.session_state["filters_seasons"] = ["Todas"]
if "filters_models" not in st.session_state:
    st.session_state["filters_models"] = ["Todas"]
if "filters_date_min" not in st.session_state:
    st.session_state["filters_date_min"] = df["match_date"].min().date()
if "filters_date_max" not in st.session_state:
    st.session_state["filters_date_max"] = df["match_date"].max().date()
if "filters_prob_range" not in st.session_state:
    st.session_state["filters_prob_range"] = (
        float(df["probability"].min()),
        float(df["probability"].max()),
    )
if "filters_odd_over_range" not in st.session_state:
    st.session_state["filters_odd_over_range"] = (
        float(df["odd_goals_over_2_5"].min()),
        float(df["odd_goals_over_2_5"].max()),
    )
if "filters_odd_under_range" not in st.session_state:
    st.session_state["filters_odd_under_range"] = (
        float(df["odd_goals_under_2_5"].min()),
        float(df["odd_goals_under_2_5"].max()),
    )

league_options = ["Todas"] + sorted(df["league_name"].dropna().unique())
season_options = ["Todas"] + sorted(df["season_id"].dropna().unique())
model_version_options = ["Todas"] + sorted(df["model_version"].dropna().unique())

prob_min, prob_max = float(df["probability"].min()), float(df["probability"].max())
odd_over_min, odd_over_max = float(df["odd_goals_over_2_5"].min()), float(df["odd_goals_over_2_5"].max())
odd_under_min, odd_under_max = float(df["odd_goals_under_2_5"].min()), float(df["odd_goals_under_2_5"].max())

with st.sidebar:
    st.subheader("Filtros de relatório (globais)")

    sel_leagues = st.multiselect(
        "Selecione a(s) liga(s)",
        league_options,
        default=st.session_state["filters_leagues"],
        key="_filters_leagues",
    )
    st.session_state["filters_leagues"] = sel_leagues

    sel_seasons = st.multiselect(
        "Selecione a(s) temporada(s)",
        season_options,
        default=st.session_state["filters_seasons"],
        key="_filters_seasons",
    )
    st.session_state["filters_seasons"] = sel_seasons

    sel_models = st.multiselect(
        "Selecione a(s) versão(ões) do modelo)",
        model_version_options,
        default=st.session_state["filters_models"],
        key="_filters_models",
    )
    st.session_state["filters_models"] = sel_models

    date_min_global = df["match_date"].min().date()
    date_max_global = df["match_date"].max().date()

    date_min_sel = st.date_input(
        "Data mínima",
        value=st.session_state["filters_date_min"],
        min_value=date_min_global,
        max_value=date_max_global,
        key="_filters_date_min",
    )
    date_max_sel = st.date_input(
        "Data máxima",
        value=st.session_state["filters_date_max"],
        min_value=date_min_global,
        max_value=date_max_global,
        key="_filters_date_max",
    )
    st.session_state["filters_date_min"] = date_min_sel
    st.session_state["filters_date_max"] = date_max_sel

    probability_range_sel = st.slider(
        "Probability (min, max)",
        min_value=prob_min,
        max_value=prob_max,
        value=st.session_state["filters_prob_range"],
        step=0.01,
        key="_filters_prob_range",
    )
    st.session_state["filters_prob_range"] = probability_range_sel

    odd_over_range_sel = st.slider(
        "Odd Over 2.5 (min, max)",
        min_value=odd_over_min,
        max_value=odd_over_max,
        value=st.session_state["filters_odd_over_range"],
        step=0.01,
        key="_filters_odd_over_range",
    )
    st.session_state["filters_odd_over_range"] = odd_over_range_sel

    odd_under_range_sel = st.slider(
        "Odd Under 2.5 (min, max)",
        min_value=odd_under_min,
        max_value=odd_under_max,
        value=st.session_state["filters_odd_under_range"],
        step=0.01,
        key="_filters_odd_under_range",
    )
    st.session_state["filters_odd_under_range"] = odd_under_range_sel


# =========================
# FILTERS (BASE / NO ODD INTERSECTION)
# =========================
selected_leagues = st.session_state["filters_leagues"]
selected_seasons = st.session_state["filters_seasons"]
selected_model_versions = st.session_state["filters_models"]
min_date = st.session_state["filters_date_min"]
max_date = st.session_state["filters_date_max"]
probability_range = st.session_state["filters_prob_range"]
odd_over_range = st.session_state["filters_odd_over_range"]
odd_under_range = st.session_state["filters_odd_under_range"]

# Categorical
if "Todas" in selected_leagues or len(selected_leagues) == 0:
    filtro_league = True
else:
    filtro_league = df["league_name"].isin(selected_leagues)

if "Todas" in selected_seasons or len(selected_seasons) == 0:
    filtro_season = True
else:
    filtro_season = df["season_id"].isin(selected_seasons)

if "Todas" in selected_model_versions or len(selected_model_versions) == 0:
    filtro_model_version = True
else:
    filtro_model_version = df["model_version"].isin(selected_model_versions)

# Continuous (global)
filtro_data = (df["match_date"] >= pd.Timestamp(min_date)) & (df["match_date"] <= pd.Timestamp(max_date))
filtro_probability = (df["probability"] >= probability_range[0]) & (df["probability"] <= probability_range[1])

df_base = df[
    filtro_league
    & filtro_season
    & filtro_model_version
    & filtro_data
    & filtro_probability
].copy()

if df_base.empty:
    st.info("Nenhum dado disponível para análise de ROI (após filtros globais).")
    st.stop()

# Drop NAs (as you requested) – we will also report how many were excluded
needed_cols = ["probability", "result", "match_date", "odd_goals_over_2_5", "odd_goals_under_2_5"]
missing_cols = [c for c in needed_cols if c not in df_base.columns]
if missing_cols:
    st.error(f"Colunas obrigatórias não encontradas: {missing_cols}")
    st.stop()

na_prob = int(df_base["probability"].isna().sum())
na_result = int(df_base["result"].isna().sum())
na_date = int(df_base["match_date"].isna().sum())
na_odd_over = int(df_base["odd_goals_over_2_5"].isna().sum())
na_odd_under = int(df_base["odd_goals_under_2_5"].isna().sum())

# We exclude NaNs in probability/result/date for any analysis
df_base = df_base.dropna(subset=["probability", "result", "match_date"]).copy()

# Market-specific odds filters + exclude NaNs in that market odd
filtro_odd_over_range = (df_base["odd_goals_over_2_5"] >= odd_over_range[0]) & (df_base["odd_goals_over_2_5"] <= odd_over_range[1])
filtro_odd_under_range = (df_base["odd_goals_under_2_5"] >= odd_under_range[0]) & (df_base["odd_goals_under_2_5"] <= odd_under_range[1])

df_over = df_base.dropna(subset=["odd_goals_over_2_5"]).copy()
df_under = df_base.dropna(subset=["odd_goals_under_2_5"]).copy()

df_over = df_over[filtro_odd_over_range.loc[df_over.index]].copy()
df_under = df_under[filtro_odd_under_range.loc[df_under.index]].copy()

with st.expander("📌 Diagnóstico de NaNs excluídos", expanded=False):
    st.write(
        pd.DataFrame(
            {
                "Coluna": ["probability", "result", "match_date", "odd_goals_over_2_5", "odd_goals_under_2_5"],
                "NaNs (antes do drop)": [na_prob, na_result, na_date, na_odd_over, na_odd_under],
            }
        )
    )
    st.caption(
        "Obs: probability/result/match_date são removidos da base global. "
        "As odds são removidas separadamente por mercado (Over remove NaN em odd_over; Under remove NaN em odd_under)."
    )

# =========================
# HELPERS (CONSISTENT THRESHOLDS)
# =========================
def _threshold_grid(start: float, stop: float, step: float, descending: bool = False) -> np.ndarray:
    """
    Builds an exact decimal grid (0.00, 0.01, ..., 1.00) using integer arithmetic to avoid float drift.
    """
    scale = int(round(1 / step))
    a = int(round(start * scale))
    b = int(round(stop * scale))
    if not descending:
        vals = np.arange(a, b + 1, 1, dtype=int)
    else:
        vals = np.arange(a, b - 1, -1, dtype=int)
    return (vals / scale).astype(float)

def _odd_grid(start: float, stop: float, step: float) -> np.ndarray:
    scale = int(round(1 / step))
    a = int(round(start * scale))
    b = int(round(stop * scale))
    vals = np.arange(a, b + 1, 1, dtype=int)
    return (vals / scale).astype(float)

def analisar_conf_odd_matriz(df_in: pd.DataFrame, tipo: str = "over"):
    """
    IMPORTANT: uses exact thresholds from integer-based grids to match manual filters.
    Over cell:  prob >= conf_min AND odd_over >= odd_min
    Under cell: prob <= conf_min AND odd_under >= odd_min   (conf_min here is a 'cap' on prob(OVER))
    """
    odd_over_col = "odd_goals_over_2_5"
    odd_under_col = "odd_goals_under_2_5"

    if tipo == "over":
        faixas_conf = _threshold_grid(0.00, 1.00, 0.01, descending=False)  # inclusive
    else:
        faixas_conf = _threshold_grid(1.00, 0.00, 0.01, descending=True)   # inclusive

    faixas_odd = _odd_grid(1.10, 2.20, 0.01)

    map_result = {"under": 0, "over": 1}
    df2 = df_in.copy()
    df2["result_norm"] = df2["result"].astype(str).str.strip().str.lower()
    df2["y_true"] = df2["result_norm"].map(map_result)
    df2 = df2[df2["y_true"].notna()].copy()

    linhas = []
    for thr_conf in faixas_conf:
        for thr_odd in faixas_odd:
            if tipo == "over":
                subset = df2[(df2["probability"] >= thr_conf) & (df2[odd_over_col] >= thr_odd)]
            else:
                subset = df2[(df2["probability"] <= thr_conf) & (df2[odd_under_col] >= thr_odd)]

            n_apostas = int(len(subset))
            if n_apostas > 0:
                if tipo == "over":
                    ganhos = np.where(subset["y_true"] == 1, subset[odd_over_col] - 1, -1)
                else:
                    ganhos = np.where(subset["y_true"] == 0, subset[odd_under_col] - 1, -1)
                roi = float(ganhos.sum() / n_apostas * 100.0)
            else:
                roi = np.nan

            linhas.append(
                {
                    "conf_min": float(np.round(thr_conf, 2)),
                    "odd_min": float(np.round(thr_odd, 2)),
                    "n": n_apostas,
                    "roi": float(np.round(roi, 2)) if n_apostas > 0 else np.nan,
                }
            )

    df_long = pd.DataFrame(linhas)

    df_long["roi_n"] = np.where(
        df_long["n"] > 0,
        df_long["roi"].astype(str) + "% (" + df_long["n"].astype(str) + ")",
        "N/A",
    )

    matriz_roi = df_long.pivot(index="conf_min", columns="odd_min", values="roi")
    matriz_n = df_long.pivot(index="conf_min", columns="odd_min", values="n")
    matriz_roi_n = df_long.pivot(index="conf_min", columns="odd_min", values="roi_n")

    return matriz_roi, matriz_n, matriz_roi_n, df_long

def plot_heatmap_text(matriz_z: pd.DataFrame, matriz_text: pd.DataFrame, titulo: str):
    st.subheader(titulo)
    fig = go.Figure(
        data=go.Heatmap(
            z=matriz_z.values,
            x=matriz_z.columns,
            y=matriz_z.index,
            text=matriz_text.values,
            texttemplate="%{text}",
            colorscale="RdYlGn",
            zmid=0,
            colorbar=dict(title="ROI (%)"),
        )
    )
    fig.update_layout(
        height=700,
        font=dict(size=20),
        xaxis=dict(title=dict(text="Odd mínima", font=dict(size=22)), tickfont=dict(size=18)),
        yaxis=dict(title=dict(text="Confiança (Over: mín | Under: máx)", font=dict(size=22)), tickfont=dict(size=18)),
    )
    st.plotly_chart(fig, use_container_width=True, key=titulo)

def calcular_roi_por_liga(df_liga: pd.DataFrame) -> float:
    odd_over_col = "odd_goals_over_2_5"
    odd_under_col = "odd_goals_under_2_5"
    resultado = df_liga["result"].astype(str).str.strip().str.lower().map({"under": 0, "over": 1})
    pred = (df_liga["probability"] >= 0.5).astype(int)
    ganhos = np.where(
        resultado == pred,
        np.where(pred == 1, df_liga[odd_over_col] - 1, df_liga[odd_under_col] - 1),
        -1,
    )
    if len(df_liga) == 0:
        return np.nan
    return float(ganhos.sum() / len(df_liga) * 100.0)

def curva_otima_from_grid(
    df_grid: pd.DataFrame,
    roi_alvo: float,
    n_min: int,
    conf_min: float,
    conf_max: float,
) -> pd.DataFrame:
    dfg = df_grid.copy()
    # ensure same rounding convention as matrix indices
    conf_min = float(np.round(conf_min, 2))
    conf_max = float(np.round(conf_max, 2))

    dfg = dfg[
        (dfg["conf_min"] >= conf_min)
        & (dfg["conf_min"] <= conf_max)
        & (dfg["n"] >= int(n_min))
        & (dfg["roi"] >= float(roi_alvo))
    ].copy()

    if dfg.empty:
        return pd.DataFrame(columns=["conf_thr", "odd_min_otima", "roi_%", "n_apostas"])

    dfg = dfg.sort_values(["conf_min", "odd_min"], ascending=[True, True])

    curva = (
        dfg.groupby("conf_min", as_index=False)
        .first()[["conf_min", "odd_min", "roi", "n"]]
        .rename(
            columns={
                "conf_min": "conf_thr",
                "odd_min": "odd_min_otima",
                "roi": "roi_%",
                "n": "n_apostas",
            }
        )
    )
    return curva

def _calc_roi_from_manual_filter(df_in: pd.DataFrame, mercado: str, conf_min: float, conf_max: float, odd_min: float):
    """
    Manual (weekly page) semantics:
    - OVER: probability in [conf_min, conf_max], odd_over >= odd_min
    - UNDER (as used on your weekly page): probability_under in [conf_min, conf_max], odd_under >= odd_min
      where probability_under = 1 - probability
    """
    df2 = df_in.copy()
    df2["result_norm"] = df2["result"].astype(str).str.strip().str.lower()
    df2["real_over"] = (df2["result_norm"] == "over").astype(int)
    df2["probability_under"] = 1.0 - df2["probability"]

    if mercado == "over":
        sel = df2[
            (df2["probability"] >= conf_min)
            & (df2["probability"] <= conf_max)
            & (df2["odd_goals_over_2_5"] >= odd_min)
        ].copy()
        sel["lucro"] = np.where(sel["real_over"] == 1, sel["odd_goals_over_2_5"] - 1, -1)
    else:
        sel = df2[
            (df2["probability_under"] >= conf_min)
            & (df2["probability_under"] <= conf_max)
            & (df2["odd_goals_under_2_5"] >= odd_min)
        ].copy()
        sel["lucro"] = np.where(sel["real_over"] == 0, sel["odd_goals_under_2_5"] - 1, -1)

    n = int(len(sel))
    if n == 0:
        return {"n": 0, "roi": np.nan, "lucro": 0.0, "df": sel}

    lucro = float(sel["lucro"].sum())
    roi = float(lucro / n * 100.0)
    return {"n": n, "roi": roi, "lucro": lucro, "df": sel}

def _matrix_cell_metrics(df_in: pd.DataFrame, tipo: str, conf_thr: float, odd_thr: float):
    """
    Pulls exact metrics using the same logic as the matrix (no conf_max, only >= or <=).
    """
    df2 = df_in.copy()
    df2["result_norm"] = df2["result"].astype(str).str.strip().str.lower()
    df2["y_true"] = df2["result_norm"].map({"under": 0, "over": 1})
    df2 = df2[df2["y_true"].notna()].copy()

    conf_thr = float(np.round(conf_thr, 2))
    odd_thr = float(np.round(odd_thr, 2))

    if tipo == "over":
        sel = df2[(df2["probability"] >= conf_thr) & (df2["odd_goals_over_2_5"] >= odd_thr)].copy()
        sel["lucro"] = np.where(sel["y_true"] == 1, sel["odd_goals_over_2_5"] - 1, -1)
    else:
        sel = df2[(df2["probability"] <= conf_thr) & (df2["odd_goals_under_2_5"] >= odd_thr)].copy()
        sel["lucro"] = np.where(sel["y_true"] == 0, sel["odd_goals_under_2_5"] - 1, -1)

    n = int(len(sel))
    if n == 0:
        return {"n": 0, "roi": np.nan, "lucro": 0.0, "df": sel}

    lucro = float(sel["lucro"].sum())
    roi = float(lucro / n * 100.0)
    return {"n": n, "roi": roi, "lucro": lucro, "df": sel}

# =========================
# PAGE BODY
# =========================
if df_over.empty and df_under.empty:
    st.info("Sem dados para OVER e UNDER após filtros (incluindo exclusão de NaNs em odds por mercado).")
    st.stop()

# Build matrices per market (no intersection)
if not df_over.empty:
    matriz_roi_over, matriz_n_over, matriz_roi_n_over, grid_over = analisar_conf_odd_matriz(df_over, tipo="over")
    plot_heatmap_text(
        matriz_roi_over,
        matriz_roi_n_over,
        "📈 OVER 2.5 — ROI (%) por confiança × odd mínima",
    )
else:
    grid_over = pd.DataFrame(columns=["conf_min", "odd_min", "n", "roi"])

if not df_under.empty:
    matriz_roi_under, matriz_n_under, matriz_roi_n_under, grid_under = analisar_conf_odd_matriz(df_under, tipo="under")
    plot_heatmap_text(
        matriz_roi_under,
        matriz_roi_n_under,
        "📉 UNDER 2.5 — ROI (%) por confiança × odd mínima",
    )
else:
    grid_under = pd.DataFrame(columns=["conf_min", "odd_min", "n", "roi"])

# ROI & entries by league (use base global, but excluding NaNs in result/prob/date)
count_entradas = df_base.groupby("league_name")["probability"].count()
roi_liga = df_base.groupby("league_name").apply(calcular_roi_por_liga)
league_stats = pd.DataFrame({"N_entradas": count_entradas, "ROI_total": roi_liga}).reset_index()

st.subheader("ROI e N Entradas por Liga")

roi_colors = ["green" if x >= 0 else "red" for x in league_stats["ROI_total"]]
bar_color = "rgba(0, 102, 255, 0.4)"

fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=league_stats["league_name"],
        y=league_stats["N_entradas"],
        name="Número de Entradas",
        yaxis="y2",
        marker_color=bar_color,
        text=league_stats["N_entradas"],
        textposition="auto",
    )
)
fig.add_trace(
    go.Scatter(
        x=league_stats["league_name"],
        y=league_stats["ROI_total"],
        mode="lines+markers",
        name="ROI (%)",
        yaxis="y1",
        marker=dict(color=roi_colors, size=12),
        line=dict(color="gray", width=2),
        text=[f"{x:.2f}%" for x in league_stats["ROI_total"]],
        textposition="top center",
    )
)
fig.update_layout(
    title="ROI e Número de Entradas por Liga",
    xaxis_tickangle=-45,
    xaxis=dict(title="Liga"),
    yaxis=dict(
        title=dict(text="ROI (%)", font=dict(color="green")),
        tickfont=dict(color="green"),
        side="left",
        showgrid=False,
        zeroline=False,
    ),
    yaxis2=dict(
        title=dict(text="Número de Entradas", font=dict(color="blue")),
        tickfont=dict(color="blue"),
        overlaying="y",
        side="right",
        showgrid=False,
        zeroline=False,
    ),
    legend=dict(x=0.5, y=1.1, orientation="h", xanchor="center"),
    bargap=0.2,
    height=500,
)
st.plotly_chart(fig, use_container_width=True, key="roi_barras_liga")

# =========================
# CURVA ÓTIMA (from grid)
# =========================
st.header("Curva Ótima de Odd mínima x Confiança (Goals 2.5)")

col_roi, col_nmin = st.columns(2)
with col_roi:
    roi_alvo = st.number_input(
        "ROI alvo para a curva (%)",
        min_value=-100.0,
        max_value=200.0,
        value=10.0,
        step=1.0,
    )
with col_nmin:
    n_min = st.number_input(
        "N mínimo de entradas por ponto de confiança",
        min_value=1,
        max_value=2000,
        value=10,
        step=1,
    )

conf_min = float(probability_range[0])
conf_max = float(probability_range[1])

st.caption(
    f"Usando confiança mínima {conf_min:.2f} e máxima {conf_max:.2f} (do filtro de probability da barra lateral)."
)

tabs = st.tabs(["Over 2.5", "Under 2.5"])

with tabs[0]:
    st.subheader("Curva Ótima - Over 2.5")
    if df_over.empty:
        st.warning("Sem dados para Over após filtros globais + odd Over (e exclusão de NaNs).")
    else:
        curva_over = curva_otima_from_grid(grid_over, float(roi_alvo), int(n_min), conf_min, conf_max)

        if curva_over.empty:
            st.warning("Nenhum ponto da curva atinge o ROI alvo com esse N mínimo para Over 2.5.")
        else:
            fig_over_curve = px.line(
                curva_over,
                x="odd_min_otima",
                y="conf_thr",
                markers=True,
            )
            fig_over_curve.update_layout(
                xaxis_title="Odd mínima ótima",
                yaxis_title="Confiança mínima (prob >= conf_thr)",
                height=350,
            )
            st.plotly_chart(fig_over_curve, use_container_width=True)
            st.dataframe(curva_over)

with tabs[1]:
    st.subheader("Curva Ótima - Under 2.5")
    if df_under.empty:
        st.warning("Sem dados para Under após filtros globais + odd Under (e exclusão de NaNs).")
    else:
        curva_under = curva_otima_from_grid(grid_under, float(roi_alvo), int(n_min), conf_min, conf_max)

        if curva_under.empty:
            st.warning("Nenhum ponto da curva atinge o ROI alvo com esse N mínimo para Under 2.5.")
        else:
            fig_under_curve = px.line(
                curva_under,
                x="odd_min_otima",
                y="conf_thr",
                markers=True,
            )
            fig_under_curve.update_layout(
                xaxis_title="Odd mínima ótima",
                yaxis_title="Teto de prob. OVER (prob <= conf_thr)",
                height=350,
            )
            st.plotly_chart(fig_under_curve, use_container_width=True)
            st.dataframe(curva_under)

# =========================
# CONSISTENCY DEBUGGER (Matrix vs Weekly-like filter)
# =========================
st.header("🧪 Validador de consistência (Matriz vs Filtro Semanal)")

col_a, col_b, col_c = st.columns(3)
with col_a:
    dbg_mercado = st.selectbox("Mercado", options=["over", "under"], index=0, key="_dbg_mkt")
with col_b:
    dbg_conf = st.number_input("conf_min (ex.: 0.42)", min_value=0.0, max_value=1.0, value=0.42, step=0.01, key="_dbg_conf")
with col_c:
    dbg_odd = st.number_input("odd_min (ex.: 2.10)", min_value=1.0, max_value=10.0, value=2.10, step=0.01, key="_dbg_odd")

dbg_conf = float(np.round(dbg_conf, 2))
dbg_odd = float(np.round(dbg_odd, 2))

st.caption(
    "O filtro semanal usa conf_min..conf_max e, para UNDER, usa probability_under = 1 - probability. "
    "A matriz NÃO usa conf_max e, para UNDER, usa prob_over <= conf_thr. "
    "Aqui eu comparo explicitamente os dois jeitos, e também mostro diferenças de partidas."
)

# Weekly-like, using df_base (same global filters) + excluding NaNs in both odds (to mimic 'eligible odds')
# Important: weekly page currently doesn't apply sidebar odd ranges; here we compare strictly by the pair (conf,odd)
weekly_df = df_base.dropna(subset=["odd_goals_over_2_5", "odd_goals_under_2_5"]).copy()

weekly_metrics = _calc_roi_from_manual_filter(
    weekly_df,
    mercado=dbg_mercado,
    conf_min=dbg_conf,
    conf_max=1.0,
    odd_min=dbg_odd,
)

# Matrix semantics: use df_over/df_under market base, and matrix logic
if dbg_mercado == "over":
    mat_metrics = _matrix_cell_metrics(df_over, tipo="over", conf_thr=dbg_conf, odd_thr=dbg_odd)
else:
    # For UNDER: to match weekly UNDER(conf_under >= 0.42), the equivalent cap on prob_over is <= (1 - 0.42) = 0.58
    # However, the user input dbg_conf is "conf_min" and ambiguous; here we show BOTH interpretations.
    mat_metrics_cap = _matrix_cell_metrics(df_under, tipo="under", conf_thr=dbg_conf, odd_thr=dbg_odd)
    mat_metrics_equiv = _matrix_cell_metrics(df_under, tipo="under", conf_thr=float(np.round(1.0 - dbg_conf, 2)), odd_thr=dbg_odd)

st.subheader("Resumo numérico")

if dbg_mercado == "over":
    show_df = pd.DataFrame(
        [
            {"Fonte": "Filtro Semanal (manual)", "N": weekly_metrics["n"], "ROI (%)": weekly_metrics["roi"], "Lucro (u)": weekly_metrics["lucro"]},
            {"Fonte": "Matriz (célula)", "N": mat_metrics["n"], "ROI (%)": mat_metrics["roi"], "Lucro (u)": mat_metrics["lucro"]},
        ]
    )
else:
    show_df = pd.DataFrame(
        [
            {"Fonte": "Filtro Semanal UNDER (prob_under>=conf)", "N": weekly_metrics["n"], "ROI (%)": weekly_metrics["roi"], "Lucro (u)": weekly_metrics["lucro"]},
            {"Fonte": "Matriz UNDER (prob_over<=conf_thr)", "N": mat_metrics_cap["n"], "ROI (%)": mat_metrics_cap["roi"], "Lucro (u)": mat_metrics_cap["lucro"], "conf_thr usado": dbg_conf},
            {"Fonte": "Matriz UNDER (equivalente conf_under)", "N": mat_metrics_equiv["n"], "ROI (%)": mat_metrics_equiv["roi"], "Lucro (u)": mat_metrics_equiv["lucro"], "conf_thr usado": float(np.round(1.0 - dbg_conf, 2))},
        ]
    )

st.dataframe(show_df)

st.subheader("Diferenças de partidas (IDs/keys)")

# Try to find a stable unique key
candidate_keys = [c for c in ["prediction_id", "match_id", "fixture_id", "game_id", "id"] if c in df_base.columns]
if candidate_keys:
    key_col = candidate_keys[0]
else:
    # fallback composite key (best-effort)
    key_col = None
    weekly_metrics["df"]["__key__"] = (
        weekly_metrics["df"]["match_date"].astype(str) + " | " +
        weekly_metrics["df"].get("league_name", "").astype(str) + " | " +
        weekly_metrics["df"].get("home_team", "").astype(str) + " vs " +
        weekly_metrics["df"].get("away_team", "").astype(str)
    )
    if dbg_mercado == "over":
        mat_metrics["df"]["__key__"] = (
            mat_metrics["df"]["match_date"].astype(str) + " | " +
            mat_metrics["df"].get("league_name", "").astype(str) + " | " +
            mat_metrics["df"].get("home_team", "").astype(str) + " vs " +
            mat_metrics["df"].get("away_team", "").astype(str)
        )
    else:
        mat_metrics_cap["df"]["__key__"] = (
            mat_metrics_cap["df"]["match_date"].astype(str) + " | " +
            mat_metrics_cap["df"].get("league_name", "").astype(str) + " | " +
            mat_metrics_cap["df"].get("home_team", "").astype(str) + " vs " +
            mat_metrics_cap["df"].get("away_team", "").astype(str)
        )
    key_col = "__key__"

if dbg_mercado == "over":
    w_keys = set(weekly_metrics["df"][key_col].astype(str).tolist())
    m_keys = set(mat_metrics["df"][key_col].astype(str).tolist())

    only_weekly = sorted(list(w_keys - m_keys))[:200]
    only_matrix = sorted(list(m_keys - w_keys))[:200]

    st.write(f"Somente no Filtro Semanal: **{len(w_keys - m_keys)}** (mostrando até 200)")
    st.write(f"Somente na Matriz: **{len(m_keys - w_keys)}** (mostrando até 200)")
    if only_weekly:
        st.dataframe(pd.DataFrame({"only_weekly": only_weekly}))
    if only_matrix:
        st.dataframe(pd.DataFrame({"only_matrix": only_matrix}))
else:
    # Compare weekly-under with matrix-under EQUIVALENT threshold (most fair)
    w_keys = set(weekly_metrics["df"][key_col].astype(str).tolist())
    m_keys = set(mat_metrics_equiv["df"][key_col].astype(str).tolist())

    only_weekly = sorted(list(w_keys - m_keys))[:200]
    only_matrix = sorted(list(m_keys - w_keys))[:200]

    st.write(f"Somente no Filtro Semanal (UNDER): **{len(w_keys - m_keys)}** (mostrando até 200)")
    st.write(f"Somente na Matriz (UNDER equivalente): **{len(m_keys - w_keys)}** (mostrando até 200)")
    if only_weekly:
        st.dataframe(pd.DataFrame({"only_weekly": only_weekly}))
    if only_matrix:
        st.dataframe(pd.DataFrame({"only_matrix": only_matrix}))
