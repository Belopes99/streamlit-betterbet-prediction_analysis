import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json

# Usa credenciais do secret do Streamlit Cloud
credentials_info = json.loads(st.secrets["gcp"]["key"])
credentials = service_account.Credentials.from_service_account_info(credentials_info)
project_id = credentials_info["project_id"]
client = bigquery.Client(credentials=credentials, project=project_id)

st.title('Taxa de Acerto por Faixa de Confiança')

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

prob_min, prob_max = float(df['probability'].min()), float(df['probability'].max())
odd_over_min, odd_over_max = float(df['odd_goals_over_2_5'].min()), float(df['odd_goals_over_2_5'].max())
odd_under_min, odd_under_max = float(df['odd_goals_under_2_5'].min()), float(df['odd_goals_under_2_5'].max())

with st.sidebar:
    st.subheader('Filtros de relatório')
    selected_league = st.selectbox('Selecione a liga', league_options)
    selected_season = st.selectbox('Selecione a temporada', season_options)
    selected_model_version = st.selectbox('Selecione a versão do modelo', model_version_options)
    date_min = df['match_date'].min().date()
    date_max = df['match_date'].max().date()
    min_date = st.date_input('Data mínima', value=date_min, min_value=date_min, max_value=date_max)
    max_date = st.date_input('Data máxima', value=date_max, min_value=date_min, max_value=date_max)

    probability_range = st.slider("Probability (min, max)", min_value=prob_min, max_value=prob_max,
                                  value=(prob_min, prob_max), step=0.01)

    odd_over_range = st.slider("Odd Over 2.5 (min, max)", min_value=odd_over_min, max_value=odd_over_max,
                               value=(odd_over_min, odd_over_max), step=0.01)

    odd_under_range = st.slider("Odd Under 2.5 (min, max)", min_value=odd_under_min, max_value=odd_under_max,
                                value=(odd_under_min, odd_under_max), step=0.01)

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


def analisar_por_faixa(df, tipo="over"):
    faixas = np.arange(0.5, 1.05, 0.05)
    resultados = []
    map_result = {'under': 0, 'over': 1}

    df = df.copy()
    df['result_norm'] = df['result'].astype(str).str.strip().str.lower()
    df['y_true'] = df['result_norm'].map(map_result)
    df['y_pred'] = (df['probability'] >= 0.5).astype(int)
    df = df[df['y_true'].notna()]

    for thr in faixas:
        if tipo == 'over':
            subset = df[df['probability'] >= thr]
            acertos = (subset['y_true'] == 1).sum()
        else:
            subset = df[df['probability'] <= (1 - thr)]
            acertos = (subset['y_true'] == 0).sum()
        total = len(subset)
        taxa_acerto = (acertos / total * 100) if total > 0 else np.nan
        resultados.append({
            "Confiança mínima": f">= {thr:.2f}" if tipo == "over" else f"<= {1-thr:.2f}",
            "N apostas": total,
            "Taxa de acerto (%)": round(taxa_acerto, 2)
        })
    return pd.DataFrame(resultados)


if not df_filtered.empty:
    tabela_over = analisar_por_faixa(df_filtered, tipo="over")
    tabela_under = analisar_por_faixa(df_filtered, tipo="under")

    st.subheader("📈 Taxa de Acerto por Faixa de Confiança (OVER 2.5)")
    st.dataframe(tabela_over)

    st.subheader("📉 Taxa de Acerto por Faixa de Confiança (UNDER 2.5)")
    st.dataframe(tabela_under)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Taxa de Acerto por Faixa (Over)")
        fig_over = go.Figure()
        fig_over.add_trace(go.Scatter(
            x=tabela_over['Confiança mínima'], y=tabela_over['Taxa de acerto (%)'],
            mode='lines+markers', name='Taxa de Acerto'))
        fig_over.add_trace(go.Bar(
            x=tabela_over['Confiança mínima'], y=tabela_over['N apostas'],
            name='Número de Apostas', yaxis='y2', opacity=0.3))
        fig_over.update_layout(
            yaxis=dict(title='Taxa de Acerto (%)', range=[0, 100]),
            yaxis2=dict(title='N de Apostas', overlaying='y', side='right'),
            height=400
        )
        st.plotly_chart(fig_over, use_container_width=True)

    with col2:
        st.subheader("Taxa de Acerto por Faixa (Under)")
        fig_under = go.Figure()
        fig_under.add_trace(go.Scatter(
            x=tabela_under['Confiança mínima'], y=tabela_under['Taxa de acerto (%)'],
            mode='lines+markers', name='Taxa de Acerto'))
        fig_under.add_trace(go.Bar(
            x=tabela_under['Confiança mínima'], y=tabela_under['N apostas'],
            name='Número de Apostas', yaxis='y2', opacity=0.3))
        fig_under.update_layout(
            yaxis=dict(title='Taxa de Acerto (%)', range=[0, 100]),
            yaxis2=dict(title='N de Apostas', overlaying='y', side='right'),
            height=400
        )
        st.plotly_chart(fig_under, use_container_width=True)
else:
    st.warning("Nenhum dado disponível com os filtros selecionados.")
