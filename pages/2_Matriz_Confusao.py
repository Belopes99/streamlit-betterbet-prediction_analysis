import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import plotly.express as px
import plotly.graph_objects as go
import json

# Lê credenciais do secret
credentials_info = json.loads(st.secrets["gcp"]["key"])
credentials = service_account.Credentials.from_service_account_info(credentials_info)
project_id = credentials_info["project_id"]
client = bigquery.Client(credentials=credentials, project=project_id)

st.title('Matriz de Confusão e Métricas')

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

league_options = ['Todas'] + sorted(df['league_name'].dropna().unique())
season_options = ['Todas'] + sorted(df['season_id'].dropna().unique())
model_version_options = ['Todas'] + sorted(df['model_version'].dropna().unique())

prob_min, prob_max = float(df['probability'].min()), float(df['probability'].max())
odd_over_min, odd_over_max = float(df['odd_goals_over_2_5'].min()), float(df['odd_goals_over_2_5'].max())
odd_under_min, odd_under_max = float(df['odd_goals_under_2_5'].min()), float(df['odd_goals_under_2_5'].max())

with st.sidebar:
    st.subheader('Filtros de relatório (globais)')

    sel_leagues = st.multiselect(
        'Selecione a(s) liga(s)',
        league_options,
        default=st.session_state["filters_leagues"],
        key="_filters_leagues",
    )
    st.session_state["filters_leagues"] = sel_leagues

    sel_seasons = st.multiselect(
        'Selecione a(s) temporada(s)',
        season_options,
        default=st.session_state["filters_seasons"],
        key="_filters_seasons",
    )
    st.session_state["filters_seasons"] = sel_seasons

    sel_models = st.multiselect(
        'Selecione a(s) versão(ões) do modelo',
        model_version_options,
        default=st.session_state["filters_models"],
        key="_filters_models",
    )
    st.session_state["filters_models"] = sel_models

    date_min_global = df['match_date'].min().date()
    date_max_global = df['match_date'].max().date()

    date_min_sel = st.date_input(
        'Data mínima',
        value=st.session_state["filters_date_min"],
        min_value=date_min_global,
        max_value=date_max_global,
        key="_filters_date_min",
    )
    date_max_sel = st.date_input(
        'Data máxima',
        value=st.session_state["filters_date_max"],
        min_value=date_min_global,
        max_value=date_max_global,
        key="_filters_date_max",
    )
    st.session_state["filters_date_min"] = date_min_sel
    st.session_state["filters_date_max"] = date_max_sel

    probability_range_sel = st.slider(
        "Probability (min, max)",
        min_value=prob_min, max_value=prob_max,
        value=st.session_state["filters_prob_range"], step=0.01,
        key="_filters_prob_range",
    )
    st.session_state["filters_prob_range"] = probability_range_sel

    odd_over_range_sel = st.slider(
        "Odd Over 2.5 (min, max)",
        min_value=odd_over_min, max_value=odd_over_max,
        value=st.session_state["filters_odd_over_range"], step=0.01,
        key="_filters_odd_over_range",
    )
    st.session_state["filters_odd_over_range"] = odd_over_range_sel

    odd_under_range_sel = st.slider(
        "Odd Under 2.5 (min, max)",
        min_value=odd_under_min, max_value=odd_under_max,
        value=st.session_state["filters_odd_under_range"], step=0.01,
        key="_filters_odd_under_range",
    )
    st.session_state["filters_odd_under_range"] = odd_under_range_sel

# Ler dos filtros globais
selected_leagues = st.session_state["filters_leagues"]
selected_seasons = st.session_state["filters_seasons"]
selected_model_versions = st.session_state["filters_models"]
min_date = st.session_state["filters_date_min"]
max_date = st.session_state["filters_date_max"]
probability_range = st.session_state["filters_prob_range"]
odd_over_range = st.session_state["filters_odd_over_range"]
odd_under_range = st.session_state["filters_odd_under_range"]

# Filtros categóricos com múltipla seleção
if 'Todas' in selected_leagues or len(selected_leagues) == 0:
    filtro_league = True
else:
    filtro_league = df['league_name'].isin(selected_leagues)

if 'Todas' in selected_seasons or len(selected_seasons) == 0:
    filtro_season = True
else:
    filtro_season = df['season_id'].isin(selected_seasons)

if 'Todas' in selected_model_versions or len(selected_model_versions) == 0:
    filtro_model_version = True
else:
    filtro_model_version = df['model_version'].isin(selected_model_versions)

# Filtros numéricos e de data
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

if not df_filtered.empty:
    map_result = {'under': 0, 'over': 1}
    df_filtered['result_norm'] = df_filtered['result'].astype(str).str.strip().str.lower()
    df_filtered['y_true'] = df_filtered['result_norm'].map(map_result)
    df_filtered['y_pred'] = (df_filtered['probability'] >= 0.5).astype(int)

    eval_df = df_filtered[df_filtered['y_true'].notna()].copy()

    if not eval_df.empty:
        cm = confusion_matrix(eval_df['y_true'], eval_df['y_pred'], labels=[0, 1])
        cm_df = pd.DataFrame(
            cm,
            index=['Real: Under (0)', 'Real: Over (1)'],
            columns=['Pred: Under (0)', 'Pred: Over (1)']
        )
        st.subheader('Matriz de Confusão')
        st.dataframe(cm_df)

        acc = accuracy_score(eval_df['y_true'], eval_df['y_pred'])
        st.write(f'**Acurácia:** {acc:.2%}')

        report = classification_report(
            eval_df['y_true'],
            eval_df['y_pred'],
            target_names=['Under (0)', 'Over (1)'],
            output_dict=True
        )
        report_df = pd.DataFrame(report).transpose()[['precision', 'recall', 'f1-score', 'support']]
        st.subheader('Métricas (Precisão, Recall, F1-score)')
        st.dataframe(report_df.round(2))

        st.subheader("Matriz de Confusão (visual)")
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            labels=dict(x="Predição", y="Real"),
            x=['Under (0)', 'Over (1)'],
            y=['Under (0)', 'Over (1)'],
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        st.subheader("Métricas (visual)")
        metricas_graf = report_df.loc[['Under (0)', 'Over (1)'], ['precision', 'recall', 'f1-score']]
        fig_bar = go.Figure()
        for m in metricas_graf.columns:
            fig_bar.add_trace(
                go.Bar(
                    name=m.capitalize(),
                    x=metricas_graf.index,
                    y=metricas_graf[m].values,
                    text=[f"{v:.2f}" for v in metricas_graf[m]],
                    textposition="auto"
                )
            )
        fig_bar.update_layout(barmode='group', yaxis=dict(range=[0, 1]), height=350)
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.warning('Nenhum jogo já finalizado para calcular matriz de confusão ou métricas.')
else:
    st.warning('Nenhum jogo filtrado para análise.')
