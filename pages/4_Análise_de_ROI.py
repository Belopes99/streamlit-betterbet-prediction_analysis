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

# Ler dos filtros globais
selected_leagues = st.session_state["filters_leagues"]
selected_seasons = st.session_state["filters_seasons"]
selected_model_versions = st.session_state["filters_models"]
min_date = st.session_state["filters_date_min"]
max_date = st.session_state["filters_date_max"]
probability_range = st.session_state["filters_prob_range"]
odd_over_range = st.session_state["filters_odd_over_range"]
odd_under_range = st.session_state["filters_odd_under_range"]

# Filtros categóricos
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

# Filtros contínuos
filtro_data = (df["match_date"] >= pd.Timestamp(min_date)) & (df["match_date"] <= pd.Timestamp(max_date))
filtro_probability = (df["probability"] >= probability_range[0]) & (df["probability"] <= probability_range[1])
filtro_odd_over = (df["odd_goals_over_2_5"] >= odd_over_range[0]) & (df["odd_goals_over_2_5"] <= odd_over_range[1])
filtro_odd_under = (df["odd_goals_under_2_5"] >= odd_under_range[0]) & (df["odd_goals_under_2_5"] <= odd_under_range[1])

df_filtered = df[
    filtro_league
    & filtro_season
    & filtro_model_version
    & filtro_data
    & filtro_probability
    & filtro_odd_over
    & filtro_odd_under
].copy()

# ===== FUNÇÕES EXISTENTES =====

def analisar_conf_odd_matriz(df, tipo="over"):
    odd_over_col = "odd_goals_over_2_5"
    odd_under_col = "odd_goals_under_2_5"

    # Panorama completo:
    # - Over: conf_min de 0.00 a 1.00 (prob >= conf_min)
    # - Under: conf_min de 1.00 a 0.00 (prob <= conf_min)
    if tipo == "over":
        faixas_conf = np.arange(0.00, 1.01, 0.01)
    else:
        faixas_conf = np.arange(1.00, -0.01, -0.01)

    faixas_odd = np.arange(1.10, 2.21, 0.01)
    linhas = []

    map_result = {"under": 0, "over": 1}
    df = df.copy()
    df["result_norm"] = df["result"].astype(str).str.strip().str.lower()
    df["y_true"] = df["result_norm"].map(map_result)
    df["y_pred"] = (df["probability"] >= 0.5).astype(int)
    df = df[df["y_true"].notna()]

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
                    ganhos = np.where(subset["y_true"] == 1, subset[odd_over_col] - 1, -1)
                else:
                    acertos = (subset["y_true"] == 0).sum()
                    ganhos = np.where(subset["y_true"] == 0, subset[odd_under_col] - 1, -1)

                taxa = acertos / n_apostas * 100
                roi = ganhos.sum() / n_apostas * 100
            else:
                taxa = np.nan
                roi = np.nan

            linhas.append(
                {
                    "conf_min": round(float(thr_conf), 3),
                    "odd_min": round(float(thr_odd), 3),
                    "n": int(n_apostas),
                    "acc": round(float(taxa), 2) if n_apostas > 0 else np.nan,
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

def plot_heatmap_text(matriz_z, matriz_text, titulo):
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
        yaxis=dict(title=dict(text="Confiança mínima", font=dict(size=22)), tickfont=dict(size=18)),
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
                ganhos = np.where(subset["y_true"] == 1, subset[odd_over_col] - 1, -1)
            else:
                acertos = (subset["y_true"] == 0).sum()
                ganhos = np.where(subset["y_true"] == 0, subset[odd_under_col] - 1, -1)

            taxa = acertos / n_apostas * 100
            lucro_total = ganhos.sum()
            roi = (lucro_total / n_apostas) * 100
        else:
            taxa = np.nan
            roi = np.nan

        resultados.append(
            {
                "Odd mínima": round(thr_odd, 2),
                "N apostas": n_apostas,
                "Taxa de acerto (%)": round(taxa, 2) if n_apostas > 0 else np.nan,
                "ROI (%)": round(roi, 2) if n_apostas > 0 else np.nan,
            }
        )

    return pd.DataFrame(resultados)

def calcular_roi_por_liga(df_liga):
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

# ===== CURVA ÓTIMA (DERIVADA DO GRID DA MATRIZ) =====

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

# ===== CORPO DA PÁGINA =====

if not df_filtered.empty:
    matriz_roi_over, matriz_n_over, matriz_roi_n_over, grid_over = analisar_conf_odd_matriz(df_filtered, tipo="over")
    matriz_roi_under, matriz_n_under, matriz_roi_n_under, grid_under = analisar_conf_odd_matriz(df_filtered, tipo="under")

    plot_heatmap_text(
        matriz_roi_over,
        matriz_roi_n_over,
        "📈 OVER 2.5 — ROI (%) por confiança × odd mínima",
    )
    plot_heatmap_text(
        matriz_roi_under,
        matriz_roi_n_under,
        "📉 UNDER 2.5 — ROI (%) por confiança × odd mínima",
    )

    st.header("ROI por faixa de Odd mínima (probabilidade fixa)")
    st.caption("Faixas de odd mínima considerando prob >= 0.5 para over e prob < 0.5 para under.")

    tabela_over = analisar_por_odd(df_filtered, tipo="over")
    tabela_under = analisar_por_odd(df_filtered, tipo="under")

    st.subheader("📈 OVER 2.5 — por odd mínima (prob >= 0.50)")
    st.dataframe(tabela_over)

    st.subheader("📉 UNDER 2.5 — por odd mínima (prob < 0.50)")
    st.dataframe(tabela_under)

    st.subheader("Gráfico de ROI por Odd mínima (Over)")
    fig_over = px.line(tabela_over, x="Odd mínima", y="ROI (%)", markers=True, text="N apostas")
    fig_over.update_traces(textposition="top center")
    fig_over.update_layout(
        width=800,
        height=400,
        yaxis_title="ROI (%)",
        xaxis_title="Odd mínima",
        font=dict(size=18),
        xaxis=dict(title=dict(text="Odd mínima", font=dict(size=22)), tickfont=dict(size=18)),
        yaxis=dict(title=dict(text="ROI (%)", font=dict(size=22)), tickfont=dict(size=18)),
    )
    st.plotly_chart(fig_over, use_container_width=True, key="line_over_odd")

    st.subheader("Gráfico de ROI por Odd mínima (Under)")
    fig_under = px.line(tabela_under, x="Odd mínima", y="ROI (%)", markers=True, text="N apostas")
    fig_under.update_traces(textposition="top center")
    fig_under.update_layout(
        width=800,
        height=400,
        yaxis_title="ROI (%)",
        xaxis_title="Odd mínima",
        font=dict(size=18),
        xaxis=dict(title=dict(text="Odd mínima", font=dict(size=22)), tickfont=dict(size=18)),
        yaxis=dict(title=dict(text="ROI (%)", font=dict(size=22)), tickfont=dict(size=18)),
    )
    st.plotly_chart(fig_under, use_container_width=True, key="line_under_odd")

    # ROI e entradas por liga
    count_entradas = df_filtered.groupby("league_name")["probability"].count()
    roi_liga = df_filtered.groupby("league_name").apply(calcular_roi_por_liga)

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

    # ===== CURVA ÓTIMA ODD x CONFIANÇA (derivada da matriz) =====
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
            value=30,
            step=1,
        )

    conf_min = float(probability_range[0])
    conf_max = float(probability_range[1])

    st.caption(
        f"Usando confiança mínima {conf_min:.2f} e máxima {conf_max:.2f} (do filtro de probability da barra lateral)."
    )

    tabs = st.tabs(["Over 2.5", "Under 2.5"])

    with tabs[0]:
        st.subheader("Curva Ótima - Over 2.5")
        curva_over = curva_otima_from_grid(grid_over, roi_alvo, int(n_min), conf_min, conf_max)

        if curva_over.empty:
            st.warning("Nenhum ponto da curva atinge o ROI alvo com esse N mínimo para Over 2.5.")
        else:
            fig_over_curve = px.line(
                curva_over,
                x="odd_min_otima",
                y="conf_thr",
                markers=True,
            )
            fig_over_curve.update_layout(
                xaxis_title="Odd mínima ótima",
                yaxis_title="Confiança mínima",
                height=350,
            )
            st.plotly_chart(fig_over_curve, use_container_width=True)
            st.dataframe(curva_over)

    with tabs[1]:
        st.subheader("Curva Ótima - Under 2.5")
        curva_under = curva_otima_from_grid(grid_under, roi_alvo, int(n_min), conf_min, conf_max)

        if curva_under.empty:
            st.warning("Nenhum ponto da curva atinge o ROI alvo com esse N mínimo para Under 2.5.")
        else:
            fig_under_curve = px.line(
                curva_under,
                x="odd_min_otima",
                y="conf_thr",
                markers=True,
            )
            fig_under_curve.update_layout(
                xaxis_title="Odd mínima ótima",
                yaxis_title="Confiança mínima",
                height=350,
            )
            st.plotly_chart(fig_under_curve, use_container_width=True)
            st.dataframe(curva_under)

else:
    st.info("Nenhum dado disponível para análise de ROI.")
