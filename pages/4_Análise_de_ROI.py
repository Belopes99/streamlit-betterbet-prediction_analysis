# =========================
# PÁGINA 1 — Análise de ROI
# (substitua o arquivo atual por este)
# =========================

import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json


# =========================
# CONFIG / AUTH / LOAD
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
# FUNÇÕES DE FILTRO (ÚNICAS / REUSÁVEIS)
# =========================
def init_global_filter_state(df_: pd.DataFrame) -> None:
    # Categóricos
    if "filters_leagues" not in st.session_state:
        st.session_state["filters_leagues"] = ["Todas"]
    if "filters_seasons" not in st.session_state:
        st.session_state["filters_seasons"] = ["Todas"]
    if "filters_models" not in st.session_state:
        st.session_state["filters_models"] = ["Todas"]

    # Datas
    if "filters_date_min" not in st.session_state:
        st.session_state["filters_date_min"] = df_["match_date"].min().date()
    if "filters_date_max" not in st.session_state:
        st.session_state["filters_date_max"] = df_["match_date"].max().date()

    # Ranges (mantidos como no seu app)
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


def render_global_sidebar(df_: pd.DataFrame) -> None:
    league_options = ["Todas"] + sorted(df_["league_name"].dropna().unique()) if "league_name" in df_.columns else ["Todas"]
    season_options = ["Todas"] + sorted(df_["season_id"].dropna().unique()) if "season_id" in df_.columns else ["Todas"]
    model_version_options = ["Todas"] + sorted(df_["model_version"].dropna().unique()) if "model_version" in df_.columns else ["Todas"]

    prob_min, prob_max = float(df_["probability"].min()), float(df_["probability"].max())
    odd_over_min, odd_over_max = float(df_["odd_goals_over_2_5"].min()), float(df_["odd_goals_over_2_5"].max())
    odd_under_min, odd_under_max = float(df_["odd_goals_under_2_5"].min()), float(df_["odd_goals_under_2_5"].max())

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

        date_min_global = df_["match_date"].min().date()
        date_max_global = df_["match_date"].max().date()

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


def apply_global_filters(
    df_: pd.DataFrame,
    *,
    market: str | None = None,
) -> pd.DataFrame:
    """
    Aplica os filtros globais de forma consistente e "correta":
    - Categóricos, data, probability_range sempre.
    - Odds: por padrão, aplica APENAS o range do mercado (over/under) quando market é informado.
            Se market=None, aplica ambos (comportamento antigo).
    """
    selected_leagues = st.session_state["filters_leagues"]
    selected_seasons = st.session_state["filters_seasons"]
    selected_model_versions = st.session_state["filters_models"]
    min_date = st.session_state["filters_date_min"]
    max_date = st.session_state["filters_date_max"]
    probability_range = st.session_state["filters_prob_range"]
    odd_over_range = st.session_state["filters_odd_over_range"]
    odd_under_range = st.session_state["filters_odd_under_range"]

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

    # Contínuos comuns
    filtro_data = (df_["match_date"] >= pd.Timestamp(min_date)) & (df_["match_date"] <= pd.Timestamp(max_date))
    filtro_probability = (df_["probability"] >= probability_range[0]) & (df_["probability"] <= probability_range[1])

    # Odds (correção principal)
    if market == "over":
        filtro_odds = (df_["odd_goals_over_2_5"] >= odd_over_range[0]) & (df_["odd_goals_over_2_5"] <= odd_over_range[1])
    elif market == "under":
        filtro_odds = (df_["odd_goals_under_2_5"] >= odd_under_range[0]) & (df_["odd_goals_under_2_5"] <= odd_under_range[1])
    else:
        filtro_odd_over = (df_["odd_goals_over_2_5"] >= odd_over_range[0]) & (df_["odd_goals_over_2_5"] <= odd_over_range[1])
        filtro_odd_under = (df_["odd_goals_under_2_5"] >= odd_under_range[0]) & (df_["odd_goals_under_2_5"] <= odd_under_range[1])
        filtro_odds = filtro_odd_over & filtro_odd_under

    out = df_[filtro_league & filtro_season & filtro_model_version & filtro_data & filtro_probability & filtro_odds].copy()
    return out


# =========================
# FUNÇÕES DE ANÁLISE
# =========================
def analisar_conf_odd_matriz(df_in: pd.DataFrame, tipo: str = "over"):
    odd_over_col = "odd_goals_over_2_5"
    odd_under_col = "odd_goals_under_2_5"

    # Panorama completo:
    # - Over: conf_min 0.00..1.00 (prob >= conf_min)
    # - Under: conf_min 1.00..0.00 (prob <= conf_min)
    if tipo == "over":
        faixas_conf = np.arange(0.00, 1.01, 0.01)
    else:
        faixas_conf = np.arange(1.00, -0.01, -0.01)

    faixas_odd = np.arange(1.10, 2.21, 0.01)
    linhas = []

    map_result = {"under": 0, "over": 1}
    dfw = df_in.copy()
    dfw["result_norm"] = dfw["result"].astype(str).str.strip().str.lower()
    dfw["y_true"] = dfw["result_norm"].map(map_result)
    dfw = dfw[dfw["y_true"].notna()]

    for thr_conf in faixas_conf:
        for thr_odd in faixas_odd:
            if tipo == "over":
                subset = dfw[(dfw["probability"] >= thr_conf) & (dfw[odd_over_col] >= thr_odd)]
            else:
                subset = dfw[(dfw["probability"] <= thr_conf) & (dfw[odd_under_col] >= thr_odd)]

            n_apostas = len(subset)

            if n_apostas > 0:
                if tipo == "over":
                    ganhos = np.where(subset["y_true"] == 1, subset[odd_over_col] - 1, -1)
                else:
                    ganhos = np.where(subset["y_true"] == 0, subset[odd_under_col] - 1, -1)

                roi = ganhos.sum() / n_apostas * 100
            else:
                roi = np.nan

            linhas.append(
                {
                    "conf_min": round(float(thr_conf), 3),
                    "odd_min": round(float(thr_odd), 3),
                    "n": int(n_apostas),
                    "roi": round(float(roi), 2) if n_apostas > 0 else np.nan,
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
    resultado = df_liga["result"].str.strip().str.lower().map({"under": 0, "over": 1})
    pred = (df_liga["probability"] >= 0.5).astype(int)
    ganhos = np.where(
        resultado == pred,
        np.where(pred == 1, df_liga[odd_over_col] - 1, df_liga[odd_under_col] - 1),
        -1,
    )
    return ganhos.sum() / len(df_liga) * 100


def curva_otima_from_grid(
    df_grid: pd.DataFrame,
    roi_alvo: float,
    n_min: int,
    conf_min: float,
    conf_max: float,
) -> pd.DataFrame:
    dfg = df_grid.copy()

    dfg = dfg[
        (dfg["conf_min"] >= conf_min)
        & (dfg["conf_min"] <= conf_max)
        & (dfg["n"] >= n_min)
        & (dfg["roi"] >= roi_alvo)
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


# =========================
# SIDEBAR + APLICAÇÃO DOS FILTROS
# =========================
init_global_filter_state(df)
render_global_sidebar(df)

# IMPORTANTE: para a matriz/curva do OVER e do UNDER, aplicamos filtros consistentes por mercado.
# Assim, evita descasamento com outras páginas/estratégias.
df_filtered_over = apply_global_filters(df, market="over")
df_filtered_under = apply_global_filters(df, market="under")

# Para o painel "ROI por Liga" (que compara pred>=0.5 usando over/under odds), mantemos o comportamento antigo:
df_filtered_global = apply_global_filters(df, market=None)


# =========================
# CORPO DA PÁGINA
# =========================
if df_filtered_over.empty and df_filtered_under.empty:
    st.info("Nenhum dado disponível para análise de ROI com os filtros atuais.")
    st.stop()

# Matrizes e grids
if not df_filtered_over.empty:
    matriz_roi_over, matriz_n_over, matriz_roi_n_over, grid_over = analisar_conf_odd_matriz(df_filtered_over, tipo="over")
else:
    matriz_roi_over = matriz_n_over = matriz_roi_n_over = grid_over = None

if not df_filtered_under.empty:
    matriz_roi_under, matriz_n_under, matriz_roi_n_under, grid_under = analisar_conf_odd_matriz(df_filtered_under, tipo="under")
else:
    matriz_roi_under = matriz_n_under = matriz_roi_n_under = grid_under = None

# Heatmaps
if matriz_roi_over is not None:
    plot_heatmap_text(
        matriz_roi_over,
        matriz_roi_n_over,
        "📈 OVER 2.5 — ROI (%) por confiança × odd mínima",
    )
else:
    st.warning("Sem dados para OVER após filtros.")

if matriz_roi_under is not None:
    plot_heatmap_text(
        matriz_roi_under,
        matriz_roi_n_under,
        "📉 UNDER 2.5 — ROI (%) por confiança × odd mínima",
    )
else:
    st.warning("Sem dados para UNDER após filtros.")

# ROI e entradas por liga (mantém funcionalidade original)
if not df_filtered_global.empty and "league_name" in df_filtered_global.columns:
    count_entradas = df_filtered_global.groupby("league_name")["probability"].count()
    roi_liga = df_filtered_global.groupby("league_name").apply(calcular_roi_por_liga)
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
else:
    st.info("Sem dados suficientes para o painel por liga com os filtros atuais.")

# Curva ótima (usa conf_min/conf_max do filtro de probability do sidebar, como antes)
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

probability_range = st.session_state["filters_prob_range"]
conf_min = float(probability_range[0])
conf_max = float(probability_range[1])
st.caption(f"Usando confiança mínima {conf_min:.2f} e máxima {conf_max:.2f} (do filtro de probability da barra lateral).")

tabs = st.tabs(["Over 2.5", "Under 2.5"])

with tabs[0]:
    st.subheader("Curva Ótima - Over 2.5")
    if grid_over is None or df_filtered_over.empty:
        st.warning("Nenhum ponto disponível (sem dados OVER).")
    else:
        curva_over = curva_otima_from_grid(grid_over, roi_alvo, int(n_min), conf_min, conf_max)
        if curva_over.empty:
            st.warning("Nenhum ponto da curva atinge o ROI alvo com esse N mínimo para Over 2.5.")
        else:
            fig_over_curve = px.line(curva_over, x="odd_min_otima", y="conf_thr", markers=True)
            fig_over_curve.update_layout(
                xaxis_title="Odd mínima ótima",
                yaxis_title="Confiança mínima (prob >= conf_thr)",
                height=350,
            )
            st.plotly_chart(fig_over_curve, use_container_width=True)
            st.dataframe(curva_over)

with tabs[1]:
    st.subheader("Curva Ótima - Under 2.5")
    if grid_under is None or df_filtered_under.empty:
        st.warning("Nenhum ponto disponível (sem dados UNDER).")
    else:
        curva_under = curva_otima_from_grid(grid_under, roi_alvo, int(n_min), conf_min, conf_max)
        if curva_under.empty:
            st.warning("Nenhum ponto da curva atinge o ROI alvo com esse N mínimo para Under 2.5.")
        else:
            fig_under_curve = px.line(curva_under, x="odd_min_otima", y="conf_thr", markers=True)
            fig_under_curve.update_layout(
                xaxis_title="Odd mínima ótima",
                yaxis_title="Teto de prob. OVER (prob <= conf_thr)",
                height=350,
            )
            st.plotly_chart(fig_under_curve, use_container_width=True)
            st.dataframe(curva_under)
