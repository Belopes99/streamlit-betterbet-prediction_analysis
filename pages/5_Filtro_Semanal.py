import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
from google.cloud import bigquery
from google.oauth2 import service_account

st.title("Análise de Lucro Semanal por Estratégia")

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
df['match_date'] = pd.to_datetime(df['match_date']).dt.tz_localize(None)

needed = [
    "probability",
    "result",
    "odd_goals_over_2_5",
    "odd_goals_under_2_5",
    "match_date"
]
for c in needed:
    if c not in df.columns:
        st.error(f"Coluna obrigatória não encontrada: {c}")
        st.stop()

df = df.dropna(subset=["probability", "result", "match_date"]).copy()
df["result_norm"] = df["result"].astype(str).str.strip().str.lower()
df["real_result"] = (df["result_norm"] == "over").astype(int)

df["probability_under"] = 1 - df["probability"]

conf_min = st.number_input("Confiança MÍNIMA", min_value=0.0, max_value=1.0, value=0.50, step=0.01)
conf_max = st.number_input("Confiança MÁXIMA", min_value=0.0, max_value=1.0, value=1.00, step=0.01)
odd_min = st.number_input("Odd mínima", min_value=1.0, value=1.30, step=0.01)
mercado = st.selectbox("Mercado apostado", options=["over", "under"], index=0)

if conf_max < conf_min:
    st.error("Confiança MÁXIMA não pode ser menor que MÍNIMA.")
    st.stop()

if mercado == "over":
    apostas = df[
        (df["probability"] >= conf_min) &
        (df["probability"] <= conf_max) &
        (df["odd_goals_over_2_5"] >= odd_min)
    ].copy()
else:
    apostas = df[
        (df["probability_under"] >= conf_min) &
        (df["probability_under"] <= conf_max) &
        (df["odd_goals_under_2_5"] >= odd_min)
    ].copy()

if apostas.empty:
    st.warning("Nenhuma aposta encontrada com esses filtros.")
    st.stop()

if mercado == "over":
    apostas["lucro"] = np.where(
        apostas["real_result"] == 1,
        apostas["odd_goals_over_2_5"] - 1,
        -1
    )
else:
    apostas["lucro"] = np.where(
        apostas["real_result"] == 0,
        apostas["odd_goals_under_2_5"] - 1,
        -1
    )

apostas["ano_semana"] = apostas["match_date"].dt.strftime("%G-%V")
resumo_semana = (
    apostas
    .groupby("ano_semana")
    .agg(
        n_apostas=("lucro", "count"),
        lucro_semana=("lucro", "sum")
    )
    .sort_index()
    .reset_index()
)
resumo_semana["lucro_acumulado"] = resumo_semana["lucro_semana"].cumsum()

st.subheader("Resumo Semanal")
st.dataframe(resumo_semana)

fig = go.Figure()
fig.add_trace(go.Bar(
    x=resumo_semana["ano_semana"],
    y=resumo_semana["lucro_semana"],
    name="Lucro da semana (u)",
    marker_color='blue'
))
fig.add_trace(go.Scatter(
    x=resumo_semana["ano_semana"],
    y=resumo_semana["lucro_acumulado"],
    mode="lines+markers",
    name="Lucro acumulado (u)",
    line=dict(color='green'),
    marker=dict(size=8)
))
fig.update_layout(
    title="Lucro semanal e acumulado",
    xaxis=dict(title="Semana (ano-semana)", tickangle=-45),
    yaxis=dict(title="Unidades (u)"),
    height=500,
    legend=dict(x=0.1, y=0.95)
)
st.plotly_chart(fig, use_container_width=True)
