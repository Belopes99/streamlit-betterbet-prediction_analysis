import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
from google.cloud import bigquery
from google.oauth2 import service_account

st.title("Análise de Lucro por Estratégia")

# Autenticação via secrets do Streamlit Cloud
credentials_info = json.loads(st.secrets["gcp"]["key"])
credentials = service_account.Credentials.from_service_account_info(credentials_info)
project_id = credentials_info["project_id"]
client = bigquery.Client(credentials=credentials, project=project_id)

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

# Garante colunas necessárias
needed = [
    "probability",
    "result",
    "odd_goals_over_2_5",
    "odd_goals_under_2_5",
    "match_date",
]
for c in needed:
    if c not in df.columns:
        st.error(f"Coluna obrigatória não encontrada: {c}")
        st.stop()

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

# Opções para filtros categóricos
if "league_name" in df.columns:
    league_options = ["Todas"] + sorted(df["league_name"].dropna().unique())
else:
    league_options = ["Todas"]

if "season_id" in df.columns:
    season_options = ["Todas"] + sorted(df["season_id"].dropna().unique())
else:
    season_options = ["Todas"]

if "model_version" in df.columns:
    model_version_options = ["Todas"] + sorted(df["model_version"].dropna().unique())
else:
    model_version_options = ["Todas"]

with st.sidebar:
    st.subheader("Filtros de amostra (globais/opcionais)")

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
        "Selecione a(s) versão(ões) de modelo",
        model_version_options,
        default=st.session_state["filters_models"],
        key="_filters_models",
    )
    st.session_state["filters_models"] = sel_models

    date_min_global = df["match_date"].min().date()
    date_max_global = df["match_date"].max().date()
    date_min_sel = st.date_input(
        "Data mínima (amostra)",
        value=st.session_state["filters_date_min"],
        min_value=date_min_global,
        max_value=date_max_global,
        key="_filters_date_min",
    )
    date_max_sel = st.date_input(
        "Data máxima (amostra)",
        value=st.session_state["filters_date_max"],
        min_value=date_min_global,
        max_value=date_max_global,
        key="_filters_date_max",
    )
    st.session_state["filters_date_min"] = date_min_sel
    st.session_state["filters_date_max"] = date_max_sel

    # Escolha da granularidade temporal
    freq_label = st.selectbox(
        "Granularidade da evolução",
        ["Diário", "Semanal", "Quinzenal", "Mensal"],
        index=1,  # padrão: Semanal
        key="_freq_evolucao",
    )

# Ler dos filtros globais
selected_leagues = st.session_state["filters_leagues"]
selected_seasons = st.session_state["filters_seasons"]
selected_model_versions = st.session_state["filters_models"]
min_date = st.session_state["filters_date_min"]
max_date = st.session_state["filters_date_max"]

# Filtros categóricos
if "league_name" in df.columns:
    if "Todas" in selected_leagues or len(selected_leagues) == 0:
        filtro_league = True
    else:
        filtro_league = df["league_name"].isin(selected_leagues)
else:
    filtro_league = True

if "season_id" in df.columns:
    if "Todas" in selected_seasons or len(selected_seasons) == 0:
        filtro_season = True
    else:
        filtro_season = df["season_id"].isin(selected_seasons)
else:
    filtro_season = True

if "model_version" in df.columns:
    if "Todas" in selected_model_versions or len(selected_model_versions) == 0:
        filtro_model_version = True
    else:
        filtro_model_version = df["model_version"].isin(selected_model_versions)
else:
    filtro_model_version = True

# Filtro por data
filtro_data = (df["match_date"] >= pd.Timestamp(min_date)) & (
    df["match_date"] <= pd.Timestamp(max_date)
)

df = df[filtro_league & filtro_season & filtro_model_version & filtro_data].copy()

df = df.dropna(subset=["probability", "result", "match_date"]).copy()
df["result_norm"] = df["result"].astype(str).str.strip().str.lower()
df["real_result"] = (df["result_norm"] == "over").astype(int)

df["probability_under"] = 1 - df["probability"]

st.subheader("Parâmetros da estratégia")

conf_min = st.number_input(
    "Confiança MÍNIMA", min_value=0.0, max_value=1.0, value=0.50, step=0.01
)
conf_max = st.number_input(
    "Confiança MÁXIMA", min_value=0.0, max_value=1.0, value=1.00, step=0.01
)
odd_min = st.number_input("Odd mínima", min_value=1.0, value=1.30, step=0.01)
mercado = st.selectbox("Mercado apostado", options=["over", "under"], index=0)

if conf_max < conf_min:
    st.error("Confiança MÁXIMA não pode ser menor que MÍNIMA.")
    st.stop()

if mercado == "over":
    apostas = df[
        (df["probability"] >= conf_min)
        & (df["probability"] <= conf_max)
        & (df["odd_goals_over_2_5"] >= odd_min)
    ].copy()
else:
    apostas = df[
        (df["probability_under"] >= conf_min)
        & (df["probability_under"] <= conf_max)
        & (df["odd_goals_under_2_5"] >= odd_min)
    ].copy()

if apostas.empty:
    st.warning("Nenhuma aposta encontrada com esses filtros.")
    st.stop()

# Lucro por aposta
if mercado == "over":
    apostas["lucro"] = np.where(
        apostas["real_result"] == 1,
        apostas["odd_goals_over_2_5"] - 1,
        -1,
    )
else:
    apostas["lucro"] = np.where(
        apostas["real_result"] == 0,
        apostas["odd_goals_under_2_5"] - 1,
        -1,
    )

# Escolha da coluna de período conforme granularidade
if freq_label == "Diário":
    apostas["periodo"] = apostas["match_date"].dt.to_period("D").astype(str)
elif freq_label == "Semanal":
    apostas["periodo"] = apostas["match_date"].dt.strftime("%G-%V")
elif freq_label == "Quinzenal":
    # ano + número da quinzena (1 ou 2) dentro do mês
    ano = apostas["match_date"].dt.year.astype(str)
    mes = apostas["match_date"].dt.month.astype(str).str.zfill(2)
    quinzena = np.where(apostas["match_date"].dt.day <= 15, "1", "2")
    apostas["periodo"] = ano + "-" + mes + "-Q" + quinzena
else:  # Mensal
    apostas["periodo"] = apostas["match_date"].dt.to_period("M").astype(str)

# Agrupamento por período com ROI
resumo_periodo = (
    apostas.groupby("periodo")
    .agg(
        n_apostas=("lucro", "count"),
        lucro_periodo=("lucro", "sum"),
    )
    .sort_index()
    .reset_index()
)
# ROI por período: lucro médio por unidade apostada (%)
resumo_periodo["roi_periodo_%"] = (
    resumo_periodo["lucro_periodo"] / resumo_periodo["n_apostas"]
) * 100
resumo_periodo["lucro_acumulado"] = resumo_periodo["lucro_periodo"].cumsum()
resumo_periodo["roi_acumulado_%"] = (
    resumo_periodo["lucro_acumulado"] / resumo_periodo["n_apostas"].cumsum()
) * 100

st.subheader(f"Resumo por período ({freq_label})")
st.dataframe(resumo_periodo)

fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=resumo_periodo["periodo"],
        y=resumo_periodo["lucro_periodo"],
        name="Lucro do período (u)",
        marker_color="blue",
    )
)
fig.add_trace(
    go.Scatter(
        x=resumo_periodo["periodo"],
        y=resumo_periodo["lucro_acumulado"],
        mode="lines+markers",
        name="Lucro acumulado (u)",
        line=dict(color="green"),
        marker=dict(size=8),
        yaxis="y2",
    )
)

fig.update_layout(
    title=f"Lucro e Lucro Acumulado por Período ({freq_label})",
    xaxis=dict(title="Período", tickangle=-45),
    yaxis=dict(title="Lucro do período (u)"),
    yaxis2=dict(
        title="Lucro acumulado (u)",
        overlaying="y",
        side="right",
    ),
    height=500,
    legend=dict(x=0.1, y=0.95),
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("ROI por período (%)")
fig_roi = go.Figure()
fig_roi.add_trace(
    go.Bar(
        x=resumo_periodo["periodo"],
        y=resumo_periodo["roi_periodo_%"],
        name="ROI do período (%)",
        marker_color="orange",
    )
)
fig_roi.add_trace(
    go.Scatter(
        x=resumo_periodo["periodo"],
        y=resumo_periodo["roi_acumulado_%"],
        mode="lines+markers",
        name="ROI acumulado (%)",
        line=dict(color="red"),
        marker=dict(size=8),
        yaxis="y2",
    )
)
fig_roi.update_layout(
    xaxis=dict(title="Período", tickangle=-45),
    yaxis=dict(title="ROI do período (%)"),
    yaxis2=dict(
        title="ROI acumulado (%)",
        overlaying="y",
        side="right",
    ),
    height=500,
    legend=dict(x=0.1, y=0.95),
)
st.plotly_chart(fig_roi, use_container_width=True)
