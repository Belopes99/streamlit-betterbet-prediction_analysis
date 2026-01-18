# =========================
# PÁGINA 2 — Filtro Semanal
# (substitua o arquivo atual por este)
# =========================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
from google.cloud import bigquery
from google.oauth2 import service_account


# =========================
# CONFIG / AUTH / LOAD
# =========================
st.title("Análise de Lucro por Estratégia")

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
def run_query(sql: str) -> pd.DataFrame:
    return client.query(sql).to_dataframe()

df = run_query(QUERY)
df["match_date"] = pd.to_datetime(df["match_date"]).dt.tz_localize(None)

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


# =========================
# FUNÇÕES DE FILTRO (MESMA LÓGICA DA PÁGINA DE ROI)
# =========================
def init_global_filter_state(df_: pd.DataFrame) -> None:
    if "filters_leagues" not in st.session_state:
        st.session_state["filters_leagues"] = ["Todas"]
    if "filters_seasons" not in st.session_state:
        st.session_state["filters_seasons"] = ["Todas"]
    if "filters_models" not in st.session_state:
        st.session_state["filters_models"] = ["Todas"]
    if "filters_date_min" not in st.session_state:
        st.session_state["filters_date_min"] = df_["match_date"].min().date()
    if "filters_date_max" not in st.session_state:
        st.session_state["filters_date_max"] = df_["match_date"].max().date()

    # Para deixar 1:1 com a ROI page, também expomos probability_range e odds ranges globais aqui,
    # mas eles são opcionais via toggle "usar filtros de odds/prob globais".
    if "filters_prob_range" not in st.session_state:
        st.session_state["filters_prob_range"] = (
            float(df_["probability"].min()),
            float(df_["probability"].max()),
        )
    if "filters_odd_over_range" not in st.session_state:
        st.session_state["filters_odd_over_range"] = (
            float(df_["odd_goals_over_2_5"].min()),
            float(df_["odd_goals_over_2_5"].max()),
        )
    if "filters_odd_under_range" not in st.session_state:
        st.session_state["filters_odd_under_range"] = (
            float(df_["odd_goals_under_2_5"].min()),
            float(df_["odd_goals_under_2_5"].max()),
        )


def render_sidebar(df_: pd.DataFrame) -> dict:
    league_options = ["Todas"] + sorted(df_["league_name"].dropna().unique()) if "league_name" in df_.columns else ["Todas"]
    season_options = ["Todas"] + sorted(df_["season_id"].dropna().unique()) if "season_id" in df_.columns else ["Todas"]
    model_version_options = ["Todas"] + sorted(df_["model_version"].dropna().unique()) if "model_version" in df_.columns else ["Todas"]

    prob_min, prob_max = float(df_["probability"].min()), float(df_["probability"].max())
    odd_over_min, odd_over_max = float(df_["odd_goals_over_2_5"].min()), float(df_["odd_goals_over_2_5"].max())
    odd_under_min, odd_under_max = float(df_["odd_goals_under_2_5"].min()), float(df_["odd_goals_under_2_5"].max())

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

        date_min_global = df_["match_date"].min().date()
        date_max_global = df_["match_date"].max().date()
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

        freq_label = st.selectbox(
            "Granularidade da evolução",
            ["Diário", "Semanal", "Quinzenal", "Mensal"],
            index=1,
            key="_freq_evolucao",
        )

        st.markdown("---")
        st.subheader("Compatibilidade com página de ROI")

        usar_prob_range = st.checkbox(
            "Usar filtro global de Probability (min, max) igual ao ROI",
            value=False,
            key="_usar_prob_range",
        )
        if usar_prob_range:
            probability_range_sel = st.slider(
                "Probability (min, max)",
                min_value=prob_min,
                max_value=prob_max,
                value=st.session_state["filters_prob_range"],
                step=0.01,
                key="_filters_prob_range",
            )
            st.session_state["filters_prob_range"] = probability_range_sel

        usar_odds_ranges = st.checkbox(
            "Usar filtros globais de Odds (ranges) iguais ao ROI",
            value=False,
            key="_usar_odds_ranges",
        )
        if usar_odds_ranges:
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

    return {
        "freq_label": freq_label,
        "usar_prob_range": usar_prob_range,
        "usar_odds_ranges": usar_odds_ranges,
    }


def apply_global_filters(
    df_: pd.DataFrame,
    *,
    market: str | None,
    usar_prob_range: bool,
    usar_odds_ranges: bool,
) -> pd.DataFrame:
    selected_leagues = st.session_state["filters_leagues"]
    selected_seasons = st.session_state["filters_seasons"]
    selected_model_versions = st.session_state["filters_models"]
    min_date = st.session_state["filters_date_min"]
    max_date = st.session_state["filters_date_max"]

    # Categóricos
    if "league_name" in df_.columns:
        filtro_league = True if ("Todas" in selected_leagues or len(selected_leagues) == 0) else df_["league_name"].isin(selected_leagues)
    else:
        filtro_league = True

    if "season_id" in df_.columns:
        filtro_season = True if ("Todas" in selected_seasons or len(selected_seasons) == 0) else df_["season_id"].isin(selected_seasons)
    else:
        filtro_season = True

    if "model_version" in df_.columns:
        filtro_model_version = True if ("Todas" in selected_model_versions or len(selected_model_versions) == 0) else df_["model_version"].isin(selected_model_versions)
    else:
        filtro_model_version = True

    filtro_data = (df_["match_date"] >= pd.Timestamp(min_date)) & (df_["match_date"] <= pd.Timestamp(max_date))

    mask = filtro_league & filtro_season & filtro_model_version & filtro_data

    if usar_prob_range:
        pr = st.session_state["filters_prob_range"]
        mask = mask & (df_["probability"] >= pr[0]) & (df_["probability"] <= pr[1])

    if usar_odds_ranges:
        odd_over_range = st.session_state["filters_odd_over_range"]
        odd_under_range = st.session_state["filters_odd_under_range"]
        if market == "over":
            mask = mask & (df_["odd_goals_over_2_5"] >= odd_over_range[0]) & (df_["odd_goals_over_2_5"] <= odd_over_range[1])
        elif market == "under":
            mask = mask & (df_["odd_goals_under_2_5"] >= odd_under_range[0]) & (df_["odd_goals_under_2_5"] <= odd_under_range[1])
        else:
            # fallback: ambos
            mask = mask & (df_["odd_goals_over_2_5"] >= odd_over_range[0]) & (df_["odd_goals_over_2_5"] <= odd_over_range[1])
            mask = mask & (df_["odd_goals_under_2_5"] >= odd_under_range[0]) & (df_["odd_goals_under_2_5"] <= odd_under_range[1])

    return df_[mask].copy()


# =========================
# UI / FILTROS
# =========================
init_global_filter_state(df)
sidebar_cfg = render_sidebar(df)

# limpeza / colunas derivadas
df = df.dropna(subset=["probability", "result", "match_date"]).copy()
df["result_norm"] = df["result"].astype(str).str.strip().str.lower()
df["real_result"] = (df["result_norm"] == "over").astype(int)
df["probability_under"] = 1 - df["probability"]

st.subheader("Parâmetros da estratégia")

conf_min = st.number_input("Confiança MÍNIMA", min_value=0.0, max_value=1.0, value=0.50, step=0.01)
conf_max = st.number_input("Confiança MÁXIMA", min_value=0.0, max_value=1.0, value=1.00, step=0.01)
odd_min = st.number_input("Odd mínima", min_value=1.0, value=1.30, step=0.01)
mercado = st.selectbox("Mercado apostado", options=["over", "under"], index=0)

if conf_max < conf_min:
    st.error("Confiança MÁXIMA não pode ser menor que MÍNIMA.")
    st.stop()

# aplica filtros globais consistentes (mesma lógica “correta” da página de ROI)
df_filtrado = apply_global_filters(
    df,
    market=mercado,
    usar_prob_range=sidebar_cfg["usar_prob_range"],
    usar_odds_ranges=sidebar_cfg["usar_odds_ranges"],
)

# aplica corte da estratégia
if mercado == "over":
    apostas = df_filtrado[
        (df_filtrado["probability"] >= conf_min)
        & (df_filtrado["probability"] <= conf_max)
        & (df_filtrado["odd_goals_over_2_5"] >= odd_min)
    ].copy()
else:
    apostas = df_filtrado[
        (df_filtrado["probability_under"] >= conf_min)
        & (df_filtrado["probability_under"] <= conf_max)
        & (df_filtrado["odd_goals_under_2_5"] >= odd_min)
    ].copy()

if apostas.empty:
    st.warning("Nenhuma aposta encontrada com esses filtros.")
    st.stop()

# lucro por aposta
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

freq_label = sidebar_cfg["freq_label"]
if freq_label == "Diário":
    apostas["periodo"] = apostas["match_date"].dt.to_period("D").astype(str)
elif freq_label == "Semanal":
    apostas["periodo"] = apostas["match_date"].dt.strftime("%G-%V")
elif freq_label == "Quinzenal":
    ano = apostas["match_date"].dt.year.astype(str)
    mes = apostas["match_date"].dt.month.astype(str).str.zfill(2)
    quinzena = np.where(apostas["match_date"].dt.day <= 15, "1", "2")
    apostas["periodo"] = ano + "-" + mes + "-Q" + quinzena
else:
    apostas["periodo"] = apostas["match_date"].dt.to_period("M").astype(str)

resumo_periodo = (
    apostas.groupby("periodo")
    .agg(
        n_apostas=("lucro", "count"),
        lucro_periodo=("lucro", "sum"),
    )
    .sort_index()
    .reset_index()
)

resumo_periodo["roi_periodo_%"] = (resumo_periodo["lucro_periodo"] / resumo_periodo["n_apostas"]) * 100
resumo_periodo["lucro_acumulado"] = resumo_periodo["lucro_periodo"].cumsum()
resumo_periodo["roi_acumulado_%"] = (resumo_periodo["lucro_acumulado"] / resumo_periodo["n_apostas"].cumsum()) * 100

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
    yaxis2=dict(title="Lucro acumulado (u)", overlaying="y", side="right"),
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
    yaxis2=dict(title="ROI acumulado (%)", overlaying="y", side="right"),
    height=500,
    legend=dict(x=0.1, y=0.95),
)
st.plotly_chart(fig_roi, use_container_width=True)

# pequenos KPIs finais
st.subheader("KPIs do corte")
kpi_cols = st.columns(4)
kpi_cols[0].metric("N apostas", f"{len(apostas)}")
kpi_cols[1].metric("Unidades", f"{apostas['lucro'].sum():.2f}")
kpi_cols[2].metric("ROI (%)", f"{(apostas['lucro'].mean()*100):.2f}")
kpi_cols[3].metric("Hit-rate (%)", f"{(apostas['lucro'].gt(0).mean()*100):.2f}")
