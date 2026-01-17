import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
from google.cloud import bigquery
from google.oauth2 import service_account

st.title("Análise de Lucro por Estratégia (com Matriz ROI)")

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
league_options = ["Todas"] + sorted(df["league_name"].dropna().unique()) if "league_name" in df.columns else ["Todas"]
season_options = ["Todas"] + sorted(df["season_id"].dropna().unique()) if "season_id" in df.columns else ["Todas"]
model_version_options = ["Todas"] + sorted(df["model_version"].dropna().unique()) if "model_version" in df.columns else ["Todas"]

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

    freq_label = st.selectbox(
        "Granularidade da evolução",
        ["Diário", "Semanal", "Quinzenal", "Mensal"],
        index=1,  # padrão: Semanal
        key="_freq_evolucao",
    )

# Ler filtros globais
selected_leagues = st.session_state["filters_leagues"]
selected_seasons = st.session_state["filters_seasons"]
selected_model_versions = st.session_state["filters_models"]
min_date = st.session_state["filters_date_min"]
max_date = st.session_state["filters_date_max"]

# Filtros categóricos
if "league_name" in df.columns and not ("Todas" in selected_leagues or len(selected_leagues) == 0):
    filtro_league = df["league_name"].isin(selected_leagues)
else:
    filtro_league = True

if "season_id" in df.columns and not ("Todas" in selected_seasons or len(selected_seasons) == 0):
    filtro_season = df["season_id"].isin(selected_seasons)
else:
    filtro_season = True

if "model_version" in df.columns and not ("Todas" in selected_model_versions or len(selected_model_versions) == 0):
    filtro_model_version = df["model_version"].isin(selected_model_versions)
else:
    filtro_model_version = True

# Filtro por data
filtro_data = (df["match_date"] >= pd.Timestamp(min_date)) & (df["match_date"] <= pd.Timestamp(max_date))

df = df[filtro_league & filtro_season & filtro_model_version & filtro_data].copy()

# Base limpa (mesmas premissas do semanal)
df = df.dropna(subset=["probability", "result", "match_date", "odd_goals_over_2_5", "odd_goals_under_2_5"]).copy()
df["result_norm"] = df["result"].astype(str).str.strip().str.lower()

# MESMA premissa do semanal: over=1, qualquer outra coisa vira 0
df["real_result"] = (df["result_norm"] == "over").astype(int)

# Probabilidade para under (mesma premissa do semanal)
df["probability_under"] = 1.0 - df["probability"]

st.subheader("Parâmetros da estratégia")

conf_min = st.number_input("Confiança MÍNIMA", min_value=0.0, max_value=1.0, value=0.50, step=0.01)
conf_max = st.number_input("Confiança MÁXIMA", min_value=0.0, max_value=1.0, value=1.00, step=0.01)
odd_min = st.number_input("Odd mínima", min_value=1.0, value=1.30, step=0.01)
mercado = st.selectbox("Mercado apostado", options=["over", "under"], index=0)

if conf_max < conf_min:
    st.error("Confiança MÁXIMA não pode ser menor que MÍNIMA.")
    st.stop()

# ===== Funções auxiliares =====

def add_period_column(df_in: pd.DataFrame, freq_label: str) -> pd.DataFrame:
    d = df_in.copy()
    if freq_label == "Diário":
        d["periodo"] = d["match_date"].dt.to_period("D").astype(str)
    elif freq_label == "Semanal":
        d["periodo"] = d["match_date"].dt.strftime("%G-%V")
    elif freq_label == "Quinzenal":
        ano = d["match_date"].dt.year.astype(str)
        mes = d["match_date"].dt.month.astype(str).str.zfill(2)
        quinzena = np.where(d["match_date"].dt.day <= 15, "1", "2")
        d["periodo"] = ano + "-" + mes + "-Q" + quinzena
    else:
        d["periodo"] = d["match_date"].dt.to_period("M").astype(str)
    return d

def apply_strategy_filter(df_in: pd.DataFrame, mercado: str, conf_min: float, conf_max: float, odd_min: float) -> pd.DataFrame:
    d = df_in.copy()
    if mercado == "over":
        sel = (
            (d["probability"] >= conf_min)
            & (d["probability"] <= conf_max)
            & (d["odd_goals_over_2_5"] >= odd_min)
        )
    else:
        sel = (
            (d["probability_under"] >= conf_min)
            & (d["probability_under"] <= conf_max)
            & (d["odd_goals_under_2_5"] >= odd_min)
        )
    return d[sel].copy()

def add_profit_column(df_in: pd.DataFrame, mercado: str) -> pd.DataFrame:
    d = df_in.copy()
    if mercado == "over":
        d["lucro"] = np.where(d["real_result"] == 1, d["odd_goals_over_2_5"] - 1, -1)
    else:
        d["lucro"] = np.where(d["real_result"] == 0, d["odd_goals_under_2_5"] - 1, -1)
    return d

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
        font=dict(size=18),
        xaxis=dict(title=dict(text="Odd mínima", font=dict(size=20)), tickfont=dict(size=14)),
        yaxis=dict(title=dict(text="Confiança mínima (mesmas premissas do filtro)", font=dict(size=20)), tickfont=dict(size=14)),
    )
    st.plotly_chart(fig, use_container_width=True, key=titulo)

def build_roi_matrix_from_weekly_premises(
    df_in: pd.DataFrame,
    mercado: str,
    conf_max_fixed: float,
    conf_step: float = 0.01,
    odd_min_start: float = 1.10,
    odd_min_end: float = 2.21,
    odd_step: float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Matriz alinhada ao Filtro Semanal:
      - Over: usa probability
      - Under: usa probability_under = 1 - probability
      - Confiança na matriz = conf_min (mínimo), e respeita conf_max_fixed como teto (mesmo input do filtro semanal)
      - Odd mínima >= odd_min
      - Lucro igual ao semanal
    Retorna: matriz_roi, matriz_n, matriz_roi_n, df_grid (long)
    """
    d = df_in.copy()

    faixas_conf = np.arange(0.00, 1.0000001, conf_step)
    faixas_odd = np.arange(odd_min_start, odd_min_end + 1e-9, odd_step)

    rows = []

    for thr_conf in faixas_conf:
        for thr_odd in faixas_odd:
            if mercado == "over":
                sel = (
                    (d["probability"] >= thr_conf)
                    & (d["probability"] <= conf_max_fixed)
                    & (d["odd_goals_over_2_5"] >= thr_odd)
                )
            else:
                sel = (
                    (d["probability_under"] >= thr_conf)
                    & (d["probability_under"] <= conf_max_fixed)
                    & (d["odd_goals_under_2_5"] >= thr_odd)
                )

            sub = d[sel]
            n = int(len(sub))

            if n > 0:
                sub2 = add_profit_column(sub, mercado)
                lucro = float(sub2["lucro"].sum())
                roi = (lucro / n) * 100.0
            else:
                lucro = np.nan
                roi = np.nan

            rows.append(
                {
                    "conf_min": round(float(thr_conf), 3),
                    "odd_min": round(float(thr_odd), 3),
                    "n": n,
                    "lucro": round(float(lucro), 4) if n > 0 else np.nan,
                    "roi": round(float(roi), 2) if n > 0 else np.nan,
                }
            )

    df_grid = pd.DataFrame(rows)

    df_grid["roi_n"] = np.where(
        df_grid["n"] > 0,
        df_grid["roi"].astype(str) + "% (" + df_grid["n"].astype(str) + ")",
        "N/A",
    )

    matriz_roi = df_grid.pivot(index="conf_min", columns="odd_min", values="roi")
    matriz_n = df_grid.pivot(index="conf_min", columns="odd_min", values="n")
    matriz_roi_n = df_grid.pivot(index="conf_min", columns="odd_min", values="roi_n")

    return matriz_roi, matriz_n, matriz_roi_n, df_grid

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
        return pd.DataFrame(columns=["conf_thr", "odd_min_otima", "roi_%", "n_apostas", "lucro_u"])

    dfg = dfg.sort_values(["conf_min", "odd_min"], ascending=[True, True])

    curva = (
        dfg.groupby("conf_min", as_index=False)
        .first()[["conf_min", "odd_min", "roi", "n", "lucro"]]
        .rename(
            columns={
                "conf_min": "conf_thr",
                "odd_min": "odd_min_otima",
                "roi": "roi_%",
                "n": "n_apostas",
                "lucro": "lucro_u",
            }
        )
    )

    return curva

# ===== Estratégia (como já era) =====
apostas = apply_strategy_filter(df, mercado, conf_min, conf_max, odd_min)

if apostas.empty:
    st.warning("Nenhuma aposta encontrada com esses filtros.")
    st.stop()

apostas = add_profit_column(apostas, mercado)
apostas = add_period_column(apostas, freq_label)

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

st.subheader(f"Resumo por período ({freq_label}) — Estratégia selecionada")
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

st.subheader("ROI por período (%) — Estratégia selecionada")
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

# ===== Matriz ROI (mesmas premissas do filtro semanal) =====
st.header("Matriz ROI (Heatmap) — Mesmas premissas do filtro semanal")

with st.expander("Configurações da Matriz/Curva (opcional)", expanded=False):
    conf_step = st.number_input("Passo de confiança (grid)", min_value=0.01, max_value=0.20, value=0.01, step=0.01)
    odd_start = st.number_input("Odd mínima inicial (grid)", min_value=1.01, max_value=10.0, value=1.10, step=0.01)
    odd_end = st.number_input("Odd mínima final (grid)", min_value=1.01, max_value=10.0, value=2.21, step=0.01)
    odd_step = st.number_input("Passo de odd (grid)", min_value=0.01, max_value=0.50, value=0.01, step=0.01)

tabs = st.tabs(["Over 2.5", "Under 2.5"])

with tabs[0]:
    st.subheader("Matriz — Over 2.5")
    matriz_roi_over, matriz_n_over, matriz_roi_n_over, grid_over = build_roi_matrix_from_weekly_premises(
        df_in=df,
        mercado="over",
        conf_max_fixed=float(conf_max),  # usa o mesmo teto do filtro semanal
        conf_step=float(conf_step),
        odd_min_start=float(odd_start),
        odd_min_end=float(odd_end),
        odd_step=float(odd_step),
    )
    plot_heatmap_text(matriz_roi_over, matriz_roi_n_over, "📈 OVER 2.5 — ROI (%) por confiança × odd mínima (premissas do semanal)")

    st.subheader("Curva ótima derivada da matriz — Over 2.5")
    col_roi, col_n = st.columns(2)
    with col_roi:
        roi_alvo = st.number_input("ROI alvo (%)", min_value=-100.0, max_value=200.0, value=10.0, step=1.0, key="roi_alvo_over")
    with col_n:
        n_min = st.number_input("N mínimo por ponto", min_value=1, max_value=5000, value=10, step=1, key="nmin_over")

    curva_over = curva_otima_from_grid(
        grid_over,
        roi_alvo=float(roi_alvo),
        n_min=int(n_min),
        conf_min=0.0,
        conf_max=1.0,
    )

    if curva_over.empty:
        st.warning("Nenhum ponto atende ROI alvo + N mínimo (Over).")
    else:
        fig_over_curve = px.line(curva_over, x="odd_min_otima", y="conf_thr", markers=True)
        fig_over_curve.update_layout(
            xaxis_title="Odd mínima ótima",
            yaxis_title="Confiança mínima (prob OVER >= conf_thr e <= conf_max)",
            height=350,
        )
        st.plotly_chart(fig_over_curve, use_container_width=True)
        st.dataframe(curva_over)

with tabs[1]:
    st.subheader("Matriz — Under 2.5")
    matriz_roi_under, matriz_n_under, matriz_roi_n_under, grid_under = build_roi_matrix_from_weekly_premises(
        df_in=df,
        mercado="under",
        conf_max_fixed=float(conf_max),  # mesmo teto do filtro semanal (aplicado em probability_under)
        conf_step=float(conf_step),
        odd_min_start=float(odd_start),
        odd_min_end=float(odd_end),
        odd_step=float(odd_step),
    )
    plot_heatmap_text(matriz_roi_under, matriz_roi_n_under, "📉 UNDER 2.5 — ROI (%) por confiança × odd mínima (premissas do semanal)")

    st.subheader("Curva ótima derivada da matriz — Under 2.5")
    col_roi, col_n = st.columns(2)
    with col_roi:
        roi_alvo_u = st.number_input("ROI alvo (%)", min_value=-100.0, max_value=200.0, value=10.0, step=1.0, key="roi_alvo_under")
    with col_n:
        n_min_u = st.number_input("N mínimo por ponto", min_value=1, max_value=5000, value=10, step=1, key="nmin_under")

    curva_under = curva_otima_from_grid(
        grid_under,
        roi_alvo=float(roi_alvo_u),
        n_min=int(n_min_u),
        conf_min=0.0,
        conf_max=1.0,
    )

    if curva_under.empty:
        st.warning("Nenhum ponto atende ROI alvo + N mínimo (Under).")
    else:
        fig_under_curve = px.line(curva_under, x="odd_min_otima", y="conf_thr", markers=True)
        fig_under_curve.update_layout(
            xaxis_title="Odd mínima ótima",
            yaxis_title="Confiança mínima (prob UNDER >= conf_thr e <= conf_max)",
            height=350,
        )
        st.plotly_chart(fig_under_curve, use_container_width=True)
        st.dataframe(curva_under)
