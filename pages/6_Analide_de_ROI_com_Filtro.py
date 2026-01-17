import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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

# Drop NAs – report how many were excluded
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
    Over cell:  prob >= conf_min AND odd_over >= odd_min
    Under cell: prob <= conf_min AND odd_under >= odd_min   (conf_min here is a 'cap' on prob(OVER))
    """
    odd_over_col = "odd_goals_over_2_5"
    odd_under_col = "odd_goals_under_2_5"

    if tipo == "over":
        faixas_conf = _threshold_grid(0.00, 1.00, 0.01, descending=False)
    else:
        faixas_conf = _threshold_grid(1.00, 0.00, 0.01, descending=True)

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
                ganhos = np.where(subset["y_true"] == 1, subset[odd_over_col] - 1, -1)
            else:
                subset = df2[(df2["probability"] <= thr_conf) & (df2[odd_under_col] >= thr_odd)]
                ganhos = np.where(subset["y_true"] == 0, subset[odd_under_col] - 1, -1)

            n_apostas = int(len(subset))
            if n_apostas > 0:
                lucro = float(ganhos.sum())
                roi = float(lucro / n_apostas * 100.0)
            else:
                lucro = np.nan
                roi = np.nan

            linhas.append(
                {
                    "conf_min": float(np.round(thr_conf, 2)),
                    "odd_min": float(np.round(thr_odd, 2)),
                    "n": n_apostas,
                    "roi": float(np.round(roi, 2)) if n_apostas > 0 else np.nan,
                    "lucro_u": float(np.round(lucro, 4)) if n_apostas > 0 else np.nan,
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

# =========================
# PAGE BODY
# =========================
if df_over.empty and df_under.empty:
    st.info("Sem dados para OVER e UNDER após filtros (incluindo exclusão de NaNs em odds por mercado).")
    st.stop()

# Build matrices per market (no intersection)
if not df_over.empty:
    matriz_roi_over, matriz_n_over, matriz_roi_n_over, grid_over = analisar_conf_odd_matriz(df_over, tipo="over")
    plot_heatmap_text(matriz_roi_over, matriz_roi_n_over, "📈 OVER 2.5 — ROI (%) por confiança × odd mínima")
else:
    grid_over = pd.DataFrame(columns=["conf_min", "odd_min", "n", "roi", "lucro_u"])

if not df_under.empty:
    matriz_roi_under, matriz_n_under, matriz_roi_n_under, grid_under = analisar_conf_odd_matriz(df_under, tipo="under")
    plot_heatmap_text(matriz_roi_under, matriz_roi_n_under, "📉 UNDER 2.5 — ROI (%) por confiança × odd mínima")
else:
    grid_under = pd.DataFrame(columns=["conf_min", "odd_min", "n", "roi", "lucro_u"])

# ROI & entries by league
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
# 🔎 MATRIX SCENARIO SEARCH (TRUTH = GRID)
# =========================
st.header("🔎 Buscador de cenários na matriz (verdade = matriz)")

col1, col2, col3 = st.columns(3)
with col1:
    roi_min_busca = st.number_input("ROI mínimo (%)", min_value=-100.0, max_value=200.0, value=10.0, step=1.0)
with col2:
    n_min_busca = st.number_input("N mínimo", min_value=1, max_value=5000, value=10, step=1)
with col3:
    top_k = st.number_input("Mostrar Top K", min_value=10, max_value=5000, value=200, step=10)

ordem = st.selectbox(
    "Ordenar por",
    options=[
        "Maior ROI",
        "Maior lucro (u)",
        "Maior N",
        "Menor odd mínima",
        "Maior odd mínima",
        "Maior conf",
        "Menor conf",
    ],
    index=0,
)

tabs_busca = st.tabs(["Over 2.5", "Under 2.5"])

def _filtrar_grid_para_busca(df_grid: pd.DataFrame, mercado: str) -> pd.DataFrame:
    if df_grid is None or df_grid.empty:
        return pd.DataFrame(columns=["conf_min", "odd_min", "roi", "n", "lucro_u"])

    dfg = df_grid.copy()

    # aplica ROI/N mínimos
    dfg = dfg[(dfg["n"] >= int(n_min_busca)) & (dfg["roi"].notna()) & (dfg["roi"] >= float(roi_min_busca))].copy()

    # aplica faixa de confiança global (sidebar)
    conf_min_slider = float(np.round(probability_range[0], 2))
    conf_max_slider = float(np.round(probability_range[1], 2))
    dfg = dfg[(dfg["conf_min"] >= conf_min_slider) & (dfg["conf_min"] <= conf_max_slider)].copy()

    # aplica faixa de odd do mercado (sidebar) — verdade = matriz construída em df_over/df_under já filtrados,
    # mas reforçamos aqui para ficar explícito.
    if mercado == "over":
        dfg = dfg[(dfg["odd_min"] >= float(np.round(odd_over_range[0], 2))) & (dfg["odd_min"] <= float(np.round(odd_over_range[1], 2)))].copy()
    else:
        dfg = dfg[(dfg["odd_min"] >= float(np.round(odd_under_range[0], 2))) & (dfg["odd_min"] <= float(np.round(odd_under_range[1], 2)))].copy()

    # ordenação
    if ordem == "Maior ROI":
        dfg = dfg.sort_values(["roi", "n"], ascending=[False, False])
    elif ordem == "Maior lucro (u)":
        dfg = dfg.sort_values(["lucro_u", "n"], ascending=[False, False])
    elif ordem == "Maior N":
        dfg = dfg.sort_values(["n", "roi"], ascending=[False, False])
    elif ordem == "Menor odd mínima":
        dfg = dfg.sort_values(["odd_min", "roi"], ascending=[True, False])
    elif ordem == "Maior odd mínima":
        dfg = dfg.sort_values(["odd_min", "roi"], ascending=[False, False])
    elif ordem == "Maior conf":
        dfg = dfg.sort_values(["conf_min", "roi"], ascending=[False, False])
    else:  # Menor conf
        dfg = dfg.sort_values(["conf_min", "roi"], ascending=[True, False])

    # formato final
    dfg = dfg.rename(columns={"conf_min": "conf_thr", "odd_min": "odd_min_ref", "n": "n_apostas", "roi": "roi_%"})
    dfg = dfg[["conf_thr", "odd_min_ref", "roi_%", "n_apostas", "lucro_u"]].head(int(top_k)).copy()

    return dfg

with tabs_busca[0]:
    st.subheader("Cenários (Over) que passam ROI mínimo e N mínimo")
    tabela_over = _filtrar_grid_para_busca(grid_over, mercado="over")
    if tabela_over.empty:
        st.warning("Nenhum cenário Over na matriz atende aos critérios atuais.")
    else:
        st.dataframe(tabela_over, use_container_width=True)

with tabs_busca[1]:
    st.subheader("Cenários (Under) que passam ROI mínimo e N mínimo")
    tabela_under = _filtrar_grid_para_busca(grid_under, mercado="under")
    if tabela_under.empty:
        st.warning("Nenhum cenário Under na matriz atende aos critérios atuais.")
    else:
        st.dataframe(tabela_under, use_container_width=True)
