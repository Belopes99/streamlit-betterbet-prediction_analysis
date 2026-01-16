import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json

# Credenciais do secret Streamlit Cloud
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
def run_query(sql):
    return client.query(sql).to_dataframe()

df = run_query(QUERY)
df["match_date"] = pd.to_datetime(df["match_date"]).dt.tz_localize(None)

# ---- Inicialização de estado global de filtros ----
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

    st.session_state["filters_leagues"] = st.multiselect(
        "Selecione a(s) liga(s)",
        league_options,
        default=st.session_state["filters_leagues"],
    )

    st.session_state["filters_seasons"] = st.multiselect(
        "Selecione a(s) temporada(s)",
        season_options,
        default=st.session_state["filters_seasons"],
    )

    st.session_state["filters_models"] = st.multiselect(
        "Selecione a(s) versão(ões) do modelo",
        model_version_options,
        default=st.session_state["filters_models"],
    )

    st.session_state["filters_date_min"] = st.date_input(
        "Data mínima",
        value=st.session_state["filters_date_min"],
        min_value=df["match_date"].min().date(),
        max_value=df["match_date"].max().date(),
    )

    st.session_state["filters_date_max"] = st.date_input(
        "Data máxima",
        value=st.session_state["filters_date_max"],
        min_value=df["match_date"].min().date(),
        max_value=df["match_date"].max().date(),
    )

    st.session_state["filters_prob_range"] = st.slider(
        "Probability (min, max)",
        min_value=prob_min,
        max_value=prob_max,
        value=st.session_state["filters_prob_range"],
        step=0.01,
    )

    st.session_state["filters_odd_over_range"] = st.slider(
        "Odd Over 2.5 (min, max)",
        min_value=odd_over_min,
        max_value=odd_over_max,
        value=st.session_state["filters_odd_over_range"],
        step=0.01,
    )

    st.session_state["filters_odd_under_range"] = st.slider(
        "Odd Under 2.5 (min, max)",
        min_value=odd_under_min,
        max_value=odd_under_max,
        value=st.session_state["filters_odd_under_range"],
        step=0.01,
    )

# ---- Aplicação de filtros ----
df_filtered = df.copy()

if "Todas" not in st.session_state["filters_leagues"]:
    df_filtered = df_filtered[df_filtered["league_name"].isin(st.session_state["filters_leagues"])]

if "Todas" not in st.session_state["filters_seasons"]:
    df_filtered = df_filtered[df_filtered["season_id"].isin(st.session_state["filters_seasons"])]

if "Todas" not in st.session_state["filters_models"]:
    df_filtered = df_filtered[df_filtered["model_version"].isin(st.session_state["filters_models"])]

df_filtered = df_filtered[
    (df_filtered["match_date"] >= pd.Timestamp(st.session_state["filters_date_min"])) &
    (df_filtered["match_date"] <= pd.Timestamp(st.session_state["filters_date_max"])) &
    (df_filtered["probability"].between(*st.session_state["filters_prob_range"])) &
    (df_filtered["odd_goals_over_2_5"].between(*st.session_state["filters_odd_over_range"])) &
    (df_filtered["odd_goals_under_2_5"].between(*st.session_state["filters_odd_under_range"]))
]

# ===== MATRIZ CONF x ODD =====

def analisar_conf_odd_matriz(df, tipo="over"):
    faixas_conf = np.arange(0.0, 1.01, 0.01) if tipo == "over" else np.arange(1.0, -0.01, -0.01)
    faixas_odd = np.arange(1.10, 2.21, 0.01)

    linhas = []
    map_result = {"under": 0, "over": 1}

    df = df.copy()
    df["y_true"] = df["result"].str.lower().map(map_result)

    for conf in faixas_conf:
        for odd in faixas_odd:
            if tipo == "over":
                subset = df[(df["probability"] >= conf) & (df["odd_goals_over_2_5"] >= odd)]
                ganhos = np.where(subset["y_true"] == 1, subset["odd_goals_over_2_5"] - 1, -1)
            else:
                subset = df[(df["probability"] <= conf) & (df["odd_goals_under_2_5"] >= odd)]
                ganhos = np.where(subset["y_true"] == 0, subset["odd_goals_under_2_5"] - 1, -1)

            n = len(subset)
            roi = ganhos.sum() / n * 100 if n > 0 else np.nan

            linhas.append({
                "conf_min": round(conf, 3),
                "odd_min": round(odd, 3),
                "n": n,
                "roi": round(roi, 2) if n > 0 else np.nan,
            })

    df_long = pd.DataFrame(linhas)
    df_long["roi_n"] = np.where(
        df_long["n"] > 0,
        df_long["roi"].astype(str) + "% (" + df_long["n"].astype(str) + ")",
        "N/A",
    )

    return (
        df_long.pivot(index="conf_min", columns="odd_min", values="roi"),
        df_long.pivot(index="conf_min", columns="odd_min", values="n"),
        df_long.pivot(index="conf_min", columns="odd_min", values="roi_n"),
        df_long,
    )

def curva_otima_from_grid(df_grid, roi_alvo, n_min, conf_min, conf_max):
    df = df_grid[
        (df_grid["conf_min"].between(conf_min, conf_max)) &
        (df_grid["roi"] >= roi_alvo) &
        (df_grid["n"] >= n_min)
    ].sort_values(["conf_min", "odd_min"])

    return (
        df.groupby("conf_min", as_index=False)
        .first()
        .rename(columns={
            "conf_min": "conf_thr",
            "odd_min": "odd_min_otima",
            "roi": "roi_%",
            "n": "n_apostas",
        })
    )

# ===== EXECUÇÃO =====

if df_filtered.empty:
    st.info("Nenhum dado disponível.")
    st.stop()

matriz_roi_over, _, matriz_roi_n_over, grid_over = analisar_conf_odd_matriz(df_filtered, "over")
matriz_roi_under, _, matriz_roi_n_under, grid_under = analisar_conf_odd_matriz(df_filtered, "under")

st.header("Heatmaps de ROI")

plot_heatmap_text(matriz_roi_over, matriz_roi_n_over, "📈 OVER 2.5 — ROI (%)")
plot_heatmap_text(matriz_roi_under, matriz_roi_n_under, "📉 UNDER 2.5 — ROI (%)")

st.header("Curva Ótima de Odd mínima x Confiança")

roi_alvo = st.number_input("ROI alvo (%)", value=10.0)
n_min = st.number_input("N mínimo de apostas", value=30)

conf_min, conf_max = st.session_state["filters_prob_range"]

tabs = st.tabs(["Over 2.5", "Under 2.5"])

with tabs[0]:
    curva_over = curva_otima_from_grid(grid_over, roi_alvo, n_min, conf_min, conf_max)
    st.plotly_chart(px.line(curva_over, x="odd_min_otima", y="conf_thr", markers=True))
    st.dataframe(curva_over)

with tabs[1]:
    curva_under = curva_otima_from_grid(grid_under, roi_alvo, n_min, conf_min, conf_max)
    st.plotly_chart(px.line(curva_under, x="odd_min_otima", y="conf_thr", markers=True))
    st.dataframe(curva_under)
