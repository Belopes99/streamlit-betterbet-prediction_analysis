import streamlit as st
import pandas as pd
import plotly.express as px
from google.cloud import bigquery
from google.oauth2 import service_account
import json
from sklearn.calibration import calibration_curve
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Previsões", page_icon="📊", layout="wide")

# Lê as credenciais do segredo configurado no Streamlit Cloud
credentials_info = json.loads(st.secrets["gcp"]["key"])
credentials = service_account.Credentials.from_service_account_info(credentials_info)
project_id = credentials_info["project_id"]
client = bigquery.Client(credentials=credentials, project=project_id)

st.title('Goals 2.5 Analytics - BetterBet')

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

league_options = ['Todas'] + sorted(df['league_name'].dropna().unique())
season_options = ['Todas'] + sorted(df['season_id'].dropna().unique())
model_version_options = ['Todas'] + sorted(df['model_version'].dropna().unique())

with st.sidebar:
    st.subheader('Filtros de relatório')
    selected_league = st.selectbox('Selecione a liga', league_options)
    selected_season = st.selectbox('Selecione a temporada', season_options)
    selected_model_version = st.selectbox('Selecione a versão do modelo', model_version_options)
    date_min = df['match_date'].min().date()
    date_max = df['match_date'].max().date()
    min_date = st.date_input('Data mínima', value=date_min, min_value=date_min, max_value=date_max)
    max_date = st.date_input('Data máxima', value=date_max, min_value=date_min, max_value=date_max)

    # Filtros deslizantes numéricos
    prob_min = float(df['probability'].min())
    prob_max = float(df['probability'].max())
    probability_range = st.slider("Probability (min, max)", min_value=prob_min, max_value=prob_max, value=(prob_min, prob_max), step=0.01)
    
    odd_over_min = float(df['odd_goals_over_2_5'].min())
    odd_over_max = float(df['odd_goals_over_2_5'].max())
    odd_over_range = st.slider("Odd Over 2.5 (min, max)", min_value=odd_over_min, max_value=odd_over_max, value=(odd_over_min, odd_over_max), step=0.01)
    
    odd_under_min = float(df['odd_goals_under_2_5'].min())
    odd_under_max = float(df['odd_goals_under_2_5'].max())
    odd_under_range = st.slider("Odd Under 2.5 (min, max)", min_value=odd_under_min, max_value=odd_under_max, value=(odd_under_min, odd_under_max), step=0.01)
    
filtro_league = df['league_name'] == selected_league if selected_league != 'Todas' else True
filtro_season = df['season_id'] == selected_season if selected_season != 'Todas' else True
filtro_model_version = df['model_version'] == selected_model_version if selected_model_version != 'Todas' else True
filtro_data = (df['match_date'] >= pd.Timestamp(min_date)) & (df['match_date'] <= pd.Timestamp(max_date))
filtro_probability = (df['probability'] >= probability_range[0]) & (df['probability'] <= probability_range[1])
filtro_odd_over = (df['odd_goals_over_2_5'] >= odd_over_range[0]) & (df['odd_goals_over_2_5'] <= odd_over_range[1])
filtro_odd_under = (df['odd_goals_under_2_5'] >= odd_under_range[0]) & (df['odd_goals_under_2_5'] <= odd_under_range[1])

df_filtered = df[
    filtro_league
    & filtro_season
    & filtro_model_version
    & filtro_data
    & filtro_probability
    & filtro_odd_over
    & filtro_odd_under
].copy()

st.subheader('Relatório filtrado')
st.write(f"Exibindo jogos entre: {min_date} e {max_date}")
st.write(f"Liga selecionada: {selected_league} | Temporada selecionada: {selected_season} | Modelo: {selected_model_version}")
st.dataframe(df_filtered)

# Gráfico: quantidade de jogos por liga
st.subheader('Quantidade de jogos por liga')
if not df_filtered.empty:
    liga_counts = df_filtered['league_name'].value_counts().reset_index()
    liga_counts.columns = ['Liga', 'Quantidade']
    fig_liga = px.bar(liga_counts, x='Liga', y='Quantidade', text='Quantidade', height=300)
    st.plotly_chart(fig_liga, use_container_width=True)
else:
    st.info('Nenhuma partida nos filtros atuais para mostrar por liga.')

# Gráfico: distribuição de previsões over/under
st.subheader('Distribuição das previsões (Over/Under)')
if not df_filtered.empty and 'prediction' in df_filtered.columns:
    pred_counts = df_filtered['prediction'].value_counts().reset_index()
    pred_counts.columns = ['Previsão', 'Quantidade']
    fig_pred = px.bar(pred_counts, x='Previsão', y='Quantidade', text='Quantidade', color='Previsão', height=300)
    st.plotly_chart(fig_pred, use_container_width=True)
else:
    st.info('Nenhuma previsão disponível para mostrar.')

st.subheader("Diagrama de Calibração (Over 2.5)")

if not df_filtered.empty and "probability" in df_filtered.columns and "result" in df_filtered.columns:
    df_plot = df_filtered.copy()
    df_plot["result_norm"] = df_plot["result"].astype(str).str.strip().str.lower()
    map_result = {"under": 0, "over": 1}
    df_plot["y_true"] = df_plot["result_norm"].map(map_result)
    df_plot = df_plot[df_plot["y_true"].notna()]

    y_true_all = df_plot["y_true"].astype(int).values
    y_prob_all = df_plot["probability"].astype(float).values

    # Excluindo os últimos 15 jogos (data mais recente)
    df_excl15 = df_plot.sort_values("match_date")
    if len(df_excl15) > 15:
        df_excl15 = df_excl15.iloc[:-15]
    else:
        df_excl15 = df_excl15.iloc[0:0]  # retorna vazio se não há dados suficientes

    y_true_excl = df_excl15["y_true"].astype(int).values
    y_prob_excl = df_excl15["probability"].astype(float).values

    # Curvas de calibração
    prob_true_all, prob_pred_all = calibration_curve(y_true_all, y_prob_all, n_bins=10, strategy="uniform")
    if len(df_excl15) > 0:
        prob_true_excl, prob_pred_excl = calibration_curve(y_true_excl, y_prob_excl, n_bins=10, strategy="uniform")
    else:
        prob_true_excl, prob_pred_excl = [], []

    fig_calib = go.Figure()

    # Linha de calibração perfeita
    fig_calib.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(dash="dash", color="gray"),
        name="Calibração perfeita"
    ))

    # Curva geral
    fig_calib.add_trace(go.Scatter(
        x=prob_pred_all, y=prob_true_all,
        mode="lines+markers",
        name="Todos jogos",
        marker=dict(symbol="circle", size=8, color="#1f77b4"),
        line=dict(width=2, color="#1f77b4")
    ))

    # Curva sem os 15 últimos
    if len(df_excl15) > 0:
        fig_calib.add_trace(go.Scatter(
            x=prob_pred_excl, y=prob_true_excl,
            mode="lines+markers",
            name="Excluindo 15 últimos",
            marker=dict(symbol="square", size=8, color="orange"),
            line=dict(width=2, color="orange")
        ))

    fig_calib.update_layout(
        xaxis=dict(title="Probabilidade prevista", range=[0, 1], tick0=0, dtick=0.1),
        yaxis=dict(title="Frequência observada", range=[0, 1], tick0=0, dtick=0.1),
        title="Diagrama de Calibração - Over 2.5",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=450
    )

    st.plotly_chart(fig_calib, use_container_width=True, key="calibracao_over25")

    # (Opcional) Histograma das probabilidades previstas
    st.subheader("Distribuição das probabilidades previstas")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=y_prob_all, bins=20,
        name="Todos jogos",
        opacity=0.7,
        marker_color="#1f77b4"
    ))
    fig_hist.update_layout(
        xaxis=dict(title="Probabilidade de Over 2.5"),
        yaxis=dict(title="N jogos"),
        bargap=0.05,
        height=250
    )
    st.plotly_chart(fig_hist, use_container_width=True, key="hist_prob")
else:
    st.info("Sem dados suficientes para gerar a curva de calibração.")
