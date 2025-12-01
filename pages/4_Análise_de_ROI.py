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

# Filtros contínuos
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

# MATRIZ CONF X ODD

def analisar_conf_odd_matriz(df, tipo="over"):
    odd_over_col = "odd_goals_over_2_5"
    odd_under_col = "odd_goals_under_2_5"

    if tipo == "over":
        faixas_conf = np.arange(0.50, 1.01, 0.025)
    else:
        faixas_conf = np.arange(0.50, -0.01, -0.05)

    faixas_odd = np.arange(1.30, 2.21, 0.025)
    linhas = []

    map_result = {"under": 0, "over": 1}
    df = df.copy()
    df["result_norm"] = df["result"].astype(str).str.strip().str.lower()
    df["y_true"] = df["result_norm"].map(map_result)
    df["y_pred"] = (df["probability"] >= 0.5).astype(int)
    df = df[df['y_true'].notna()]

    for thr_conf in faixas_conf:
        for thr_odd in faixas_odd:
            if tipo == "over":
                subset = df[(df["probability"] >= thr_conf) & (df[odd_over_col] >= thr_odd)]
            else:
                subset = df[(df["probability"] <= thr_conf) & (df[odd_under_col] >= thr_odd)]

            n_apostas = len(subset)

            if n_apostas > 0:
                if tipo == "over":
                    acertos = (subset["y_true"] == 1).sum()
                    ganhos = np.where(subset["y_true"] == 1,
                                      subset[odd_over_col] - 1,
                                      -1)
                else:
                    acertos = (subset["y_true"] == 0).sum()
                    ganhos = np.where(subset["y_true"] == 0,
                                      subset[odd_under_col] - 1,
                                      -1)

                taxa = acertos / n_apostas * 100
                roi = ganhos.sum() / n_apostas * 100
            else:
                taxa = np.nan
                roi = np.nan

            linhas.append({
                "conf_min": round(thr_conf, 3),
                "odd_min": round(thr_odd, 3),
                "n": n_apostas,
                "acc": round(taxa, 2) if n_apostas > 0 else np.nan,
                "roi": round(roi, 2) if n_apostas > 0 else np.nan
            })
    df_long = pd.DataFrame(linhas)
    matriz_roi = df_long.pivot(index="conf_min", columns="odd_min", values="roi")
    matriz_n = df_long.pivot(index="conf_min", columns="odd_min", values="n")
    df_long["roi_n"] = np.where(
        df_long["n"] > 0,
        df_long["roi"].astype(str) + "% (" + df_long["n"].astype(str) + ")",
        "N/A"
    )
    matriz_roi_n = df_long.pivot(index="conf_min", columns="odd_min", values="roi_n")
    return matriz_roi, matriz_n, matriz_roi_n

def plot_heatmap_text(matriz_z, matriz_text, titulo):
    st.subheader(titulo)
    fig = go.Figure(data=go.Heatmap(
        z=matriz_z.values,
        x=matriz_z.columns,
        y=matriz_z.index,
        text=matriz_text.values,
        texttemplate="%{text}",
        colorscale='RdYlGn',
        zmid=0,
        colorbar=dict(title="ROI (%)")
    ))
    fig.update_layout(
        height=700,
        font=dict(size=20),
        xaxis=dict(
            title=dict(text='Odd mínima', font=dict(size=22)),
            tickfont=dict(size=18)
        ),
        yaxis=dict(
            title=dict(text='Confiança mínima', font=dict(size=22)),
            tickfont=dict(size=18)
        )
    )
    st.plotly_chart(fig, use_container_width=True, key=titulo)

def analisar_por_odd(df, tipo="over"):
    odd_over_col = "odd_goals_over_2_5"
    odd_under_col = "odd_goals_under_2_5"
    faixas_odd = np.arange(1.3, 2.81, 0.1)
    resultados = []

    map_result = {"under": 0, "over": 1}
    df = df.copy()
    df["result_norm"] = df["result"].astype(str).str.strip().str.lower()
    df["y_true"] = df["result_norm"].map(map_result)

    for thr_odd in faixas_odd:
        if tipo == "over":
            subset = df[(df["probability"] >= 0.5) & (df[odd_over_col] >= thr_odd)]
        else:
            subset = df[(df["probability"] < 0.5) & (df[odd_under_col] >= thr_odd)]

        n_apostas = len(subset)

        if n_apostas > 0:
            if tipo == "over":
                acertos = (subset["y_true"] == 1).sum()
                ganhos = np.where(
                    subset["y_true"] == 1,
                    subset[odd_over_col] - 1,
                    -1
                )
            else:
                acertos = (subset["y_true"] == 0).sum()
                ganhos = np.where(
                    subset["y_true"] == 0,
                    subset[odd_under_col] - 1,
                    -1
                )

            taxa = acertos / n_apostas * 100
            lucro_total = ganhos.sum()
            roi = (lucro_total / n_apostas) * 100
        else:
            taxa = np.nan
            roi = np.nan

        resultados.append({
            "Odd mínima": round(thr_odd, 2),
            "N apostas": n_apostas,
            "Taxa de acerto (%)": round(taxa, 2) if n_apostas > 0 else np.nan,
            "ROI (%)": round(roi, 2) if n_apostas > 0 else np.nan
        })

    return pd.DataFrame(resultados)

def analisar_por_edge(df, tipo="over"):
    odd_over_col = "odd_goals_over_2_5"
    odd_under_col = "odd_goals_under_2_5"
    ev_over_col = "ev_over"
    ev_under_col = "ev_under"
    faixas_ev = np.arange(0.0, 0.16, 0.01)
    resultados = []

    map_result = {"under": 0, "over": 1}
    df = df.copy()
    df["result_norm"] = df["result"].astype(str).str.strip().str.lower()
    df["y_true"] = df["result_norm"].map(map_result)

    for edge_min in faixas_ev:
        if tipo == "over":
            subset = df[df[ev_over_col] >= edge_min]
        else:
            subset = df[df[ev_under_col] >= edge_min]

        n_apostas = len(subset)

        if n_apostas > 0:
            if tipo == "over":
                acertos = (subset["y_true"] == 1).sum()
                ganhos = np.where(
                    subset["y_true"] == 1,
                    subset[odd_over_col] - 1,
                    -1
                )
            else:
                acertos = (subset["y_true"] == 0).sum()
                ganhos = np.where(
                    subset["y_true"] == 0,
                    subset[odd_under_col] - 1,
                    -1
                )
            taxa = acertos / n_apostas * 100
            roi = ganhos.sum() / n_apostas * 100
        else:
            taxa = np.nan
            roi = np.nan

        resultados.append({
            "Edge mínimo (EV)": round(edge_min, 2),
            "N apostas": n_apostas,
            "Taxa de acerto (%)": round(taxa, 2) if n_apostas > 0 else np.nan,
            "ROI (%)": round(roi, 2) if n_apostas > 0 else np.nan
        })

    return pd.DataFrame(resultados)

def analisar_ev_coerente(df, tipo="over"):
    odd_over_col = "odd_goals_over_2_5"
    odd_under_col = "odd_goals_under_2_5"
    ev_over_col = "ev_over"
    ev_under_col = "ev_under"
    faixas_ev = np.arange(0.0, 0.16, 0.01)
    resultados = []

    map_result = {"under": 0, "over": 1}
    df = df.copy()
    df = df.dropna(subset=["probability"]).copy()
    df["result_norm"] = df["result"].astype(str).str.strip().str.lower()
    df["y_true"] = df["result_norm"].map(map_result)

    for edge_min in faixas_ev:
        if tipo == "over":
            subset = df[(df["probability"] >= 0.5) & (df[ev_over_col] >= edge_min)]
        else:
            subset = df[(df["probability"] < 0.5) & (df[ev_under_col] >= edge_min)]

        n_apostas = len(subset)

        if n_apostas > 0:
            if tipo == "over":
                acertos = (subset["y_true"] == 1).sum()
                ganhos = np.where(subset["y_true"] == 1,
                                  subset[odd_over_col] - 1,
                                  -1)
            else:
                acertos = (subset["y_true"] == 0).sum()
                ganhos = np.where(subset["y_true"] == 0,
                                  subset[odd_under_col] - 1,
                                  -1)

            taxa = acertos / n_apostas * 100
            roi = ganhos.sum() / n_apostas * 100
        else:
            taxa = np.nan
            roi = np.nan

        resultados.append({
            "Edge mínimo (EV)": round(edge_min, 2),
            "N apostas": n_apostas,
            "Taxa de acerto (%)": round(taxa, 2) if n_apostas > 0 else np.nan,
            "ROI (%)": round(roi, 2) if n_apostas > 0 else np.nan
        })

    return pd.DataFrame(resultados)

def calcular_roi_por_liga(df_liga):
    odd_over_col = "odd_goals_over_2_5"
    odd_under_col = "odd_goals_under_2_5"
    resultado = df_liga['result'].str.strip().str.lower().map({'under':0,'over':1})
    pred = (df_liga['probability'] >= 0.5).astype(int)
    ganhos = np.where(
        resultado == pred,
        np.where(pred == 1, df_liga[odd_over_col] - 1, df_liga[odd_under_col] - 1),
        -1
    )
    return ganhos.sum() / len(df_liga) * 100

if not df_filtered.empty:
    matriz_roi_over, matriz_n_over, matriz_roi_n_over = analisar_conf_odd_matriz(df_filtered, tipo="over")
    matriz_roi_under, matriz_n_under, matriz_roi_n_under = analisar_conf_odd_matriz(df_filtered, tipo="under")

    plot_heatmap_text(matriz_roi_over, matriz_roi_n_over, "📈 OVER 2.5 — ROI (%) por confiança × odd mínima")
    plot_heatmap_text(matriz_roi_under, matriz_roi_n_under, "📉 UNDER 2.5 — ROI (%) por confiança × odd mínima")

    # --- Análise "por Odd mínima" abaixo dos heatmaps ---
    st.header('ROI por faixa de Odd mínima (probabilidade fixa)')
    st.caption('Faixas de odd mínima considerando prob >= 0.5 para over e prob < 0.5 para under.')

    tabela_over = analisar_por_odd(df_filtered, tipo="over")
    tabela_under = analisar_por_odd(df_filtered, tipo="under")

    st.subheader('📈 OVER 2.5 — por odd mínima (prob >= 0.50)')
    st.dataframe(tabela_over)

    st.subheader('📉 UNDER 2.5 — por odd mínima (prob < 0.50)')
    st.dataframe(tabela_under)

    # Gráficos para ROI por odd
    st.subheader('Gráfico de ROI por Odd mínima (Over)')
    fig_over = px.line(tabela_over, x='Odd mínima', y='ROI (%)', markers=True, text='N apostas')
    fig_over.update_traces(textposition='top center')
    fig_over.update_layout(
        width=800, height=400,
        yaxis_title='ROI (%)',
        xaxis_title='Odd mínima',
        font=dict(size=18),
        xaxis=dict(title=dict(text='Odd mínima', font=dict(size=22)), tickfont=dict(size=18)),
        yaxis=dict(title=dict(text='ROI (%)', font=dict(size=22)), tickfont=dict(size=18))
    )
    st.plotly_chart(fig_over, use_container_width=True, key="line_over_odd")

    st.subheader('Gráfico de ROI por Odd mínima (Under)')
    fig_under = px.line(tabela_under, x='Odd mínima', y='ROI (%)', markers=True, text='N apostas')
    fig_under.update_traces(textposition='top center')
    fig_under.update_layout(
        width=800, height=400,
        yaxis_title='ROI (%)',
        xaxis_title='Odd mínima',
        font=dict(size=18),
        xaxis=dict(title=dict(text='Odd mínima', font=dict(size=22)), tickfont=dict(size=18)),
        yaxis=dict(title=dict(text='ROI (%)', font=dict(size=22)), tickfont=dict(size=18))
    )
    st.plotly_chart(fig_under, use_container_width=True, key="line_under_odd")

    st.header('EV+ — ROI por edge mínimo (sem filtro precoce)')
    tabela_ev_over = analisar_por_edge(df_filtered, tipo="over")
    tabela_ev_under = analisar_por_edge(df_filtered, tipo="under")

    st.subheader('📈 OVER 2.5 — EV+ por edge mínimo')
    st.dataframe(tabela_ev_over)

    st.subheader('📉 UNDER 2.5 — EV+ por edge mínimo')
    st.dataframe(tabela_ev_under)

    st.subheader('Gráfico EV+ (Over)')
    fig_ev_over = px.line(tabela_ev_over, x='Edge mínimo (EV)', y='ROI (%)', markers=True, text='N apostas')
    fig_ev_over.update_traces(textposition='top center')
    fig_ev_over.update_layout(
        height=400,
        font=dict(size=18),
        xaxis=dict(title=dict(text='Edge mínimo (EV)', font=dict(size=22)), tickfont=dict(size=18)),
        yaxis=dict(title=dict(text='ROI (%)', font=dict(size=22)), tickfont=dict(size=18))
    )
    st.plotly_chart(fig_ev_over, use_container_width=True, key="ev_over")

    st.subheader('Gráfico EV+ (Under)')
    fig_ev_under = px.line(tabela_ev_under, x='Edge mínimo (EV)', y='ROI (%)', markers=True, text='N apostas')
    fig_ev_under.update_traces(textposition='top center')
    fig_ev_under.update_layout(
        height=400,
        font=dict(size=18),
        xaxis=dict(title=dict(text='Edge mínimo (EV)', font=dict(size=22)), tickfont=dict(size=18)),
        yaxis=dict(title=dict(text='ROI (%)', font=dict(size=22)), tickfont=dict(size=18))
    )
    st.plotly_chart(fig_ev_under, use_container_width=True, key="ev_under")

    st.header('EV+ Compatível com Confiança do Modelo (Prob ≥ 0.5 Over, Prob < 0.5 Under)')
    tabela_ev_over_coerente = analisar_ev_coerente(df_filtered, tipo="over")
    tabela_ev_under_coerente = analisar_ev_coerente(df_filtered, tipo="under")

    st.subheader('📈 OVER 2.5 — EV+')
    st.dataframe(tabela_ev_over_coerente)

    st.subheader('📉 UNDER 2.5 — EV+')
    st.dataframe(tabela_ev_under_coerente)

    st.subheader('Gráfico EV+ (Over)')
    fig_ev_over_c = px.line(tabela_ev_over_coerente, x='Edge mínimo (EV)', y='ROI (%)', markers=True, text='N apostas')
    fig_ev_over_c.update_traces(textposition='top center')
    fig_ev_over_c.update_layout(
        height=400,
        font=dict(size=18),
        xaxis=dict(title=dict(text='Edge mínimo (EV)', font=dict(size=22)), tickfont=dict(size=18)),
        yaxis=dict(title=dict(text='ROI (%)', font=dict(size=22)), tickfont=dict(size=18))
    )
    st.plotly_chart(fig_ev_over_c, use_container_width=True, key="ev_coerente_over")

    st.subheader('Gráfico EV+ (Under)')
    fig_ev_under_c = px.line(tabela_ev_under_coerente, x='Edge mínimo (EV)', y='ROI (%)', markers=True, text='N apostas')
    fig_ev_under_c.update_traces(textposition='top center')
    fig_ev_under_c.update_layout(
        height=400,
        font=dict(size=18),
        xaxis=dict(title=dict(text='Edge mínimo (EV)', font=dict(size=22)), tickfont=dict(size=18)),
        yaxis=dict(title=dict(text='ROI (%)', font=dict(size=22)), tickfont=dict(size=18))
    )
    st.plotly_chart(fig_ev_under_c, use_container_width=True, key="ev_coerente_under")

    count_entradas = df_filtered.groupby('league_name')['probability'].count()
    roi_liga = df_filtered.groupby('league_name').apply(calcular_roi_por_liga)

    league_stats = pd.DataFrame({
        'N_entradas': count_entradas,
        'ROI_total': roi_liga
    }).reset_index()

    st.subheader("ROI e N Entradas por Liga")

    roi_colors = ['green' if x >= 0 else 'red' for x in league_stats['ROI_total']]
    bar_color = 'rgba(0, 102, 255, 0.4)'

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=league_stats['league_name'],
        y=league_stats['N_entradas'],
        name='Número de Entradas',
        yaxis='y2',
        marker_color=bar_color,
        text=league_stats['N_entradas'],
        textposition='auto'
    ))

    fig.add_trace(go.Scatter(
        x=league_stats['league_name'],
        y=league_stats['ROI_total'],
        mode='lines+markers',
        name='ROI (%)',
        yaxis='y1',
        marker=dict(color=roi_colors, size=12),
        line=dict(color='gray', width=2),
        text=[f"{x:.2f}%" for x in league_stats['ROI_total']],
        textposition='top center'
    ))

    fig.update_layout(
        title="ROI e Número de Entradas por Liga",
        xaxis_tickangle=-45,
        xaxis=dict(title='Liga'),
        yaxis=dict(
            title=dict(text='ROI (%)', font=dict(color='green')),
            tickfont=dict(color='green'),
            side='left',
            showgrid=False,
            zeroline=False
        ),
        yaxis2=dict(
            title=dict(text='Número de Entradas', font=dict(color='blue')),
            tickfont=dict(color='blue'),
            overlaying='y',
            side='right',
            showgrid=False,
            zeroline=False
        ),
        legend=dict(x=0.5, y=1.1, orientation='h', xanchor='center'),
        bargap=0.2,
        height=500
    )

    st.plotly_chart(fig, use_container_width=True, key="roi_barras_liga")

else:
    st.info("Nenhum dado disponível para análise de ROI e entradas por liga.")
