"""AgroClima — app Streamlit de inferência do rendimento agrícola por hectare.

Três modos (apenas o modo ativo é renderizado):
1. "Onde produzir?"          variáveis de clima -> mapa de um produto
2. "O que produzir?"         clique no mapa -> ranking de produtos do município
3. "Qual será o rendimento?" variáveis informadas -> rendimento/ha previsto

Rodar:  python build_reference.py  &&  streamlit run app.py
"""

from __future__ import annotations

import gc
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
import geo_utils as geo
import model_utils as mu
from index_utils import texto_explicacao_indice

for _opt, _val in (
    ("theme.primaryColor", "#2e7d32"),
    ("theme.sidebar.backgroundColor", "#2e7d32"),
    ("theme.sidebar.secondaryBackgroundColor", "#1b5e20"),
    ("theme.sidebar.textColor", "#ffffff"),
    ("theme.sidebar.primaryColor", "#ffffff"),
    ("theme.greenColor", "#2e7d32"),
    ("theme.redColor", "#2e7d32"),
    ("theme.baseRadius", "none"),
    ("theme.buttonRadius", "none"),
):
    try:
        st._config.set_option(_opt, _val)
    except Exception:  # noqa: BLE001
        pass

st.set_page_config(page_title="AgroClima", page_icon=":material/eco:", layout="wide")


def aplicar_estilo() -> None:
    """Aplica tema verde, ícones verdes e corner-radius zero em caixas, botões e tabelas."""
    st.markdown(
        """
        <style>
        :root {
            --primary-color: #2e7d32 !important;
        }
        /* Corner radius 0 em todas as caixas, entradas, botões, alertas e tabelas */
        [data-testid="stExpander"],
        [data-testid="stExpander"] details,
        [data-testid="stExpander"] summary,
        [data-testid="stForm"],
        [data-testid="stAlert"],
        [data-testid="stMetric"],
        [data-testid="stDataFrame"],
        [data-testid="stDataFrame"] > div,
        [data-testid="stDataFrame"] canvas,
        [data-testid="stTable"],
        [data-testid="stTable"] table,
        [data-testid="stTable"] th,
        [data-testid="stTable"] td,
        [data-baseweb="select"] > div,
        [data-baseweb="input"],
        [data-baseweb="base-input"],
        [data-baseweb="popover"],
        [data-baseweb="menu"],
        [data-baseweb="button-group"],
        [data-baseweb="button-group"] button,
        [data-testid="stNumberInput"] div,
        [data-testid="stNumberInput"] input,
        [data-testid="stNumberInput"] button,
        .stButton > button,
        .stDownloadButton > button,
        .stFormSubmitButton > button {
            border-radius: 0px !important;
        }
        /* Sidebar esquerda com fundo verde (#2e7d32) e fonte clara de alto contraste */
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] > div:first-child {
            background-color: #2e7d32 !important;
            color: #ffffff !important;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] li,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] summary,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"] [data-testid="stIconMaterial"],
        [data-testid="stSidebar"] button svg {
            color: #ffffff !important;
            fill: #ffffff !important;
        }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
        [data-testid="stSidebar"] small {
            color: #e8f5e9 !important;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"],
        [data-testid="stSidebar"] [data-testid="stExpander"] details {
            background-color: #1b5e20 !important;
            border-color: rgba(255, 255, 255, 0.35) !important;
        }
        /* Ícones simples e verdes na área principal */
        [data-testid="stMain"] [data-testid="stIconMaterial"] {
            color: #2e7d32 !important;
        }
        /* Botões e segmented control em verde (substituindo o vermelho padrão) */
        .stButton > button:hover,
        .stButton > button:focus,
        .stButton > button:active,
        .stDownloadButton > button:hover,
        .stDownloadButton > button:focus,
        .stDownloadButton > button:active,
        [data-baseweb="button-group"] button:hover {
            border-color: #2e7d32 !important;
            color: #2e7d32 !important;
            box-shadow: none !important;
        }
        .stButton > button[kind="primary"],
        .stFormSubmitButton > button[kind="primary"],
        [data-baseweb="button-group"] button[kind="segmented_controlActive"],
        [data-baseweb="button-group"] button[aria-checked="true"],
        [data-baseweb="button-group"] button[aria-pressed="true"] {
            background-color: #2e7d32 !important;
            border-color: #2e7d32 !important;
            color: #ffffff !important;
        }
        .stButton > button[kind="primary"] [data-testid="stIconMaterial"],
        .stFormSubmitButton > button[kind="primary"] [data-testid="stIconMaterial"] {
            color: #ffffff !important;
        }
        .stButton > button[kind="primary"]:hover,
        .stFormSubmitButton > button[kind="primary"]:hover {
            background-color: #1b5e20 !important;
            border-color: #1b5e20 !important;
            color: #ffffff !important;
        }
        /* Sliders, barras e foco de inputs em verde */
        [data-testid="stSlider"] [role="slider"] {
            background-color: #2e7d32 !important;
            border-color: #2e7d32 !important;
            box-shadow: none !important;
        }
        [data-testid="stThumbValue"] {
            color: #2e7d32 !important;
        }
        [data-baseweb="select"] > div:focus-within,
        [data-baseweb="input"]:focus-within,
        [data-baseweb="base-input"]:focus-within {
            border-color: #2e7d32 !important;
            box-shadow: 0 0 0 1px #2e7d32 !important;
        }
        [data-testid="stProgress"] > div > div > div {
            background-color: #2e7d32 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# Recursos em cache
# =============================================================================
@st.cache_resource(show_spinner="Carregando o modelo (pode levar alguns segundos)...")
def recurso_modelo():
    """Modelo XGBoost (carregado uma única vez por processo)."""
    return mu.carregar_modelo()


@st.cache_resource(show_spinner="Carregando tabelas auxiliares...")
def recurso_refs() -> mu.Referencias:
    """Tabelas geradas pelo build_reference.py."""
    return mu.carregar_referencias()


@st.cache_resource(show_spinner="Verificando o ambiente (features, categorias e teste de fumaça)...")
def recurso_validacao() -> mu.ResultadoValidacao:
    """Executa ``validar_ambiente`` uma única vez."""
    res = mu.validar_ambiente(recurso_modelo(), recurso_refs())
    gc.collect()
    return res


@st.cache_resource(show_spinner="Carregando a malha municipal...")
def recurso_malha() -> dict:
    """GeoJSON simplificado (somente leitura)."""
    return geo.carregar_malha()


@st.cache_resource(show_spinner=False)
def recurso_bbox() -> pd.DataFrame:
    """Caixas envolventes dos municípios, para enquadrar região/estado."""
    return geo.bbox_por_municipio(recurso_malha()).set_index("codigo_ibge")


@st.cache_resource(show_spinner=False)
def recurso_nomes_malha() -> pd.DataFrame:
    """Nome de exibição de todas as feições da malha (inclusive sem histórico)."""
    return pd.read_csv(config.ARQ_NOMES_MALHA, dtype={"codigo_ibge": str}, encoding="utf-8")


@st.cache_data(show_spinner="Calculando o mapa...", max_entries=16)
def cache_mapa(produto: str, ajustes_t: tuple, valor_unico_t: tuple | None) -> pd.DataFrame:
    """Modo 1: predição em lote, cacheada por (produto, ajustes de clima)."""
    return mu.prever_mapa_produto(
        recurso_modelo(), recurso_refs(), produto, dict(ajustes_t), dict(valor_unico_t) if valor_unico_t else None
    )


@st.cache_data(show_spinner=False, max_entries=8)
def cache_figura_mapa(produto: str, ajustes_t: tuple, valor_unico_t: tuple | None, metrica: str) -> go.Figure:
    """Modo 1: figura do mapa (evita reconstruir ao alternar de modo e voltar)."""
    refs = recurso_refs()
    df = cache_mapa(produto, ajustes_t, valor_unico_t)
    nomes = recurso_nomes_malha()
    com = set(df["codigo_ibge"])
    sem = [c for c in nomes["codigo_ibge"] if c not in com]
    coluna = "indice" if metrica == "indice" else "rendimento_previsto"
    titulo = "Índice" if metrica == "indice" else refs.produtos.loc[produto, "unidade"]
    return geo.figura_mapa_predicao(
        recurso_malha(), df, coluna, titulo, sem, nomes.set_index("codigo_ibge")["nome_exibicao"].to_dict()
    )


@st.cache_data(show_spinner="Calculando o ranking...", max_entries=32)
def cache_produtos(codigos_t: tuple, ajustes_t: tuple) -> pd.DataFrame:
    """Modo 2: predição de todos os produtos válidos para um ou mais municípios."""
    return mu.prever_produtos_municipios(recurso_modelo(), recurso_refs(), list(codigos_t), dict(ajustes_t))


@st.cache_resource(show_spinner=False)
def recurso_malha_estados() -> dict:
    """GeoJSON dos 27 estados (dissolvido no build_reference.py)."""
    return geo.carregar_malha(config.ARQ_MALHA_ESTADOS)


@st.cache_resource(show_spinner=False)
def recurso_malha_regioes() -> dict:
    """GeoJSON das 5 regiões (dissolvido no build_reference.py)."""
    return geo.carregar_malha(config.ARQ_MALHA_REGIOES)


@st.cache_data(show_spinner=False, max_entries=32)
def cache_figura_selecao(
    nivel: str, selecionado: str | None, enquadramento: tuple = (None, None)
) -> tuple[go.Figure, list[list[str]]]:
    """Modo 2: mapa colorido por região no nível escolhido, com a seleção destacada.

    ``selecionado``: ``codigo_ibge`` (Cidades), sigla da UF (Estados) ou nome da região (Regiões).
    ``enquadramento``: ``(regiao, uf)`` que define centro/zoom; muda apenas pelos dropdowns e pela
    troca de nível, nunca pelo clique no mapa.
    """
    refs = recurso_refs()
    regiao, uf = enquadramento
    if uf:
        cods = refs.municipios.index[refs.municipios["uf"] == uf]
    elif regiao:
        cods = refs.municipios.index[refs.municipios["regiao"] == regiao]
    else:
        cods = []
    bbox = recurso_bbox()
    centro, zoom = geo.centro_zoom(bbox.loc[bbox.index.intersection(cods)]) if len(cods) else (
        dict(config.CENTRO_BRASIL), config.ZOOM_BRASIL)
    uirevision = f"modo2|{nivel}|{regiao}|{uf}"
    if nivel == "Estados":
        return geo.figura_mapa_estados(recurso_malha_estados(), selecionado, centro, zoom, uirevision)
    if nivel == "Regiões":
        return geo.figura_mapa_regioes(recurso_malha_regioes(), selecionado, centro, zoom, uirevision)
    return geo.figura_mapa_selecao(recurso_malha(), recurso_nomes_malha(), selecionado, centro, zoom, uirevision)


# =============================================================================
# Persistência de estado por modo (prefixos m1_, m2_, m3_)
# =============================================================================
def _copiar(wkey: str, pkey: str) -> None:
    """Callback: copia o valor do widget para a chave persistente."""
    st.session_state[pkey] = st.session_state[wkey]


def persist(pkey: str, padrao: Any, opcoes: list | None = None) -> dict:
    """Prepara um widget cujo valor sobrevive à troca de modo.

    O Streamlit descarta o estado de widgets não renderizados; por isso o valor
    fica em ``st.session_state[pkey]`` e é recolocado na chave do widget ao recriá-lo.
    Se ``opcoes`` for informado e o valor salvo não estiver nelas, usa ``padrao``.
    """
    wkey = f"_w_{pkey}"
    if pkey not in st.session_state or (opcoes is not None and st.session_state[pkey] not in opcoes):
        st.session_state[pkey] = padrao
    if wkey not in st.session_state or st.session_state[wkey] != st.session_state[pkey]:
        st.session_state[wkey] = st.session_state[pkey]
    return {"key": wkey, "on_change": _copiar, "args": (wkey, pkey)}


def definir(pkey: str, valor: Any) -> None:
    """Altera um valor persistente e o widget correspondente (usar apenas em callbacks)."""
    st.session_state[pkey] = valor
    st.session_state[f"_w_{pkey}"] = valor


# =============================================================================
# Componentes reutilizáveis
# =============================================================================
def notas_fixas() -> None:
    """Notas obrigatórias em todos os modos."""
    st.caption(f":green[:material/calendar_today:] {config.NOTA_ANO}  \n:green[:material/info:] {config.NOTA_ESTIMATIVA}")


def zerar_ajustes(prefixo: str) -> None:
    """Callback: volta todos os ajustes de clima do modo para 0."""
    for var in config.CLIMA:
        definir(f"{prefixo}_aj_{var}", 0.0)


def controles_ajustes(prefixo: str) -> tuple[tuple[str, float], ...]:
    """Sliders de ajuste de cenário (aplicados sobre o clima base de cada município)."""
    st.markdown("**Cenário de clima** — ajustes aplicados sobre o clima base (média 2015–2024) de cada município")
    cols = st.columns(3)
    ajustes = []
    for i, var in enumerate(config.CLIMA):
        _, vmin, vmax, passo, rotulo = config.AJUSTES_CLIMA[var]
        with cols[i % 3]:
            v = st.slider(rotulo, float(vmin), float(vmax), step=float(passo), **persist(f"{prefixo}_aj_{var}", 0.0))
        ajustes.append((var, float(v)))
    st.button("Zerar ajustes", icon=":material/restart_alt:", key=f"{prefixo}_btn_zerar", on_click=zerar_ajustes, args=(prefixo,))
    return tuple(ajustes)


def alerta_fora_faixa(contagem: dict[str, int], total: int | None = None) -> None:
    """Alerta de extrapolação quando o clima sai da faixa observada na base."""
    if not contagem:
        return
    partes = [
        f"{config.ROTULOS_CLIMA[v]}: {n}" + (f" de {total} municípios" if total else "") for v, n in contagem.items()
    ]
    st.warning(
        "**Clima fora da faixa observada na base** — o modelo não extrapola bem, "
        "então trate estes resultados com cautela.\n\n- " + "\n- ".join(partes),
        icon=":material/warning:",
    )


def grafico_ranking(df: pd.DataFrame, titulo: str) -> go.Figure:
    """Barras horizontais do índice relativo (nunca valores brutos de produtos diferentes)."""
    d = df.sort_values("indice", ascending=True)
    cores = d["tipo_lavoura"].map({"Temporária": "#66a61e", "Permanente": "#1b9e77"}).fillna("#888888")
    fig = go.Figure(
        go.Bar(
            x=d["indice"], y=d["produto"], orientation="h", marker_color=cores,
            text=d["indice"].round(0).astype(int), textposition="outside",
            customdata=d[["tipo_lavoura"]].to_numpy(),
            hovertemplate="%{y}<br>Índice relativo: %{x:.1f}<br>%{customdata[0]}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text=titulo, font=dict(size=14)),
        xaxis=dict(title="Índice relativo (0–100)", range=[0, 108]),
        yaxis=dict(title=None, automargin=True),
        height=max(260, 26 * len(d) + 80), margin=dict(l=10, r=10, t=40, b=40),
    )
    return fig


def tabela_ranking(df: pd.DataFrame, agregado: bool = False) -> pd.DataFrame:
    """Tabela do Modo 2: rendimento bruto com unidade (ordenada pelo índice)."""
    t = df.sort_values("indice", ascending=False).copy()
    t["unidade"] = config.UNIDADE
    cols = {"produto": "Produto", "tipo_lavoura": "Tipo de lavoura", "indice": "Índice relativo",
            "rendimento_previsto": "Rendimento previsto", "unidade": "Unidade"}
    if agregado:
        cols["n_municipios"] = "Nº de municípios"
    return t[list(cols)].rename(columns=cols)


CFG_TABELA = {
    "Índice relativo": st.column_config.ProgressColumn("Índice relativo", min_value=0, max_value=100, format="%.0f"),
    "Rendimento previsto": st.column_config.NumberColumn("Rendimento previsto", format="%.3f"),
}


# =============================================================================
# Modo 1 — Onde produzir?
# =============================================================================
def modo_onde_produzir(refs: mu.Referencias) -> None:
    """Variáveis de clima -> mapa coroplético de um produto."""
    st.subheader(":green[:material/map:] Onde produzir?")
    st.write("Escolha um produto e um cenário de clima para ver em quais municípios o rendimento estimado é maior.")
    notas_fixas()

    with st.expander("Controles", expanded=True):
        c1, c2, c3 = st.columns([1, 2, 1])
        tipos = ["Todos"] + sorted(refs.produtos["tipo_lavoura"].unique())
        tipo = c1.selectbox("Tipo de lavoura (filtra a lista)", tipos, **persist("m1_tipo", "Todos", tipos))
        prods = sorted(refs.produtos.index if tipo == "Todos" else refs.produtos.index[refs.produtos["tipo_lavoura"] == tipo])
        padrao = "Soja (em grão)" if "Soja (em grão)" in prods else prods[0]
        produto = c2.selectbox("Produto", prods, **persist("m1_produto", padrao, prods))
        c3.metric("Unidade", refs.produtos.loc[produto, "unidade"],
                  help="Os valores de rendimento não são comparáveis entre produtos.")

        opcoes_metrica = ["Índice relativo (padrão)", f"Rendimento previsto ({config.UNIDADE})"]
        metrica_txt = st.radio("Colorir o mapa por", opcoes_metrica, horizontal=True,
                               **persist("m1_metrica", opcoes_metrica[0], opcoes_metrica))
        metrica = "indice" if metrica_txt == opcoes_metrica[0] else "rendimento"

        usar_unico = st.toggle("Avançado: usar um valor único de clima para todos os municípios",
                               **persist("m1_usar_valor_unico", False))
        if usar_unico:
            medianas = refs.clima_base[config.CLIMA].median()
            cols = st.columns(3)
            valores = []
            for i, var in enumerate(config.CLIMA):
                passo, fmt = config.FORMATO_CLIMA[var]
                with cols[i % 3]:
                    v = st.number_input(config.ROTULOS_CLIMA[var], step=passo, format=fmt,
                                        **persist(f"m1_vu_{var}", round(float(medianas[var]), 3)))
                valores.append((var, float(v)))
            valor_unico_t: tuple | None = tuple(valores)
            ajustes_t: tuple = tuple((v, 0.0) for v in config.CLIMA)
            st.caption("Os valores iniciais são a mediana nacional do clima base. Os ajustes de cenário ficam desativados.")
        else:
            ajustes_t = controles_ajustes("m1")
            valor_unico_t = None

    df = cache_mapa(produto, ajustes_t, valor_unico_t)
    if df.empty:
        st.error("Nenhum município tem histórico de produção deste produto.")
        return
    alerta_fora_faixa(mu.fora_da_faixa(df, refs), len(df))

    st.markdown(f"**{produto}** — {len(df)} municípios com histórico de produção "
                f"(os demais aparecem em cinza: *sem histórico de produção*).")
    fig = cache_figura_mapa(produto, ajustes_t, valor_unico_t, metrica)
    st.plotly_chart(fig, width="stretch", config={"scrollZoom": True, "displaylogo": False}, key="m1_mapa")
    with st.expander("O que é o índice relativo?"):
        st.markdown(texto_explicacao_indice())

    st.markdown("#### Melhores municípios")
    top_n = st.number_input("Quantidade (Top N)", min_value=5, max_value=500, step=5,
                            **persist("m1_top_n", config.TOP_N_PADRAO))
    col_ord = "indice" if metrica == "indice" else "rendimento_previsto"
    top = df.sort_values(col_ord, ascending=False).head(int(top_n)).copy()
    top["unidade"] = refs.produtos.loc[produto, "unidade"]
    tabela = top[["cidade", "uf", "rendimento_previsto", "unidade", "indice"]].rename(
        columns={"cidade": "Município", "uf": "UF", "rendimento_previsto": "Rendimento previsto",
                 "unidade": "Unidade", "indice": "Índice relativo"}
    )
    tabela.insert(0, "Posição", range(1, len(tabela) + 1))
    st.dataframe(tabela, hide_index=True, width="stretch", column_config=CFG_TABELA)
    st.download_button(
        "Baixar Top N (CSV)",
        tabela.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
        file_name=f"top_{int(top_n)}_{geo.normalizar_compacto(produto)}.csv",
        mime="text/csv",
        icon=":material/download:",
        key="m1_download",
    )


# =============================================================================
# Modo 2 — O que produzir?
# =============================================================================
_CHAVE_MAPA_M2: dict[str, str] = {
    "Cidades": "m2_mapa_cidades",
    "Estados": "m2_mapa_estados",
    "Regiões": "m2_mapa_regioes",
}
_TEXTO_TOPO_M2: dict[str, str] = {
    "Cidades": "Clique em um município no mapa para ver quais produtos têm o maior índice relativo previsto.",
    "Estados": "Clique em um estado no mapa para ver quais produtos têm o maior índice relativo previsto.",
    "Regiões": "Clique em uma região no mapa para ver quais produtos têm o maior índice relativo previsto.",
}
_MSG_ORIENTACAO_M2: dict[str, str] = {
    "Cidades": "Clique em um município no mapa para ver os produtos mais propensos.",
    "Estados": "Clique em um estado no mapa para ver os produtos mais propensos.",
    "Regiões": "Clique em uma região no mapa para ver os produtos mais propensos.",
}


def _ao_clicar_mapa() -> None:
    """Callback do clique no mapa do Modo 2 (Cidades, Estados ou Regiões)."""
    nivel = st.session_state.get("m2_nivel_ativo", config.NIVEIS_M2[0])
    chave_mapa = _CHAVE_MAPA_M2.get(nivel, "m2_mapa_cidades")
    estado = st.session_state.get(chave_mapa) or st.session_state.get("m2_mapa")
    pontos = []
    try:
        pontos = estado["selection"]["points"]
    except (KeyError, TypeError):
        pontos = []
    if not pontos:  # clique em área vazia ou desseleção: mantém a seleção anterior
        return
    chave = geo.codigo_do_ponto(pontos[-1], st.session_state.get("m2_curvas"))
    if not chave:
        return
    if nivel == "Cidades":
        st.session_state.pop("m2_aviso_sem_historico", None)
        st.session_state["m2_codigo_ibge"] = chave
        st.session_state["m2_painel"] = "municipio"
        return
    res = geo.area_do_clique(nivel, chave, recurso_refs().municipios)
    if res is None:
        st.session_state["m2_aviso_sem_historico"] = chave
        return
    st.session_state.pop("m2_aviso_sem_historico", None)
    reg_base, uf_base = res
    definir("m2_regiao", reg_base)
    definir("m2_uf", uf_base if uf_base else "(Todos)")
    st.session_state["m2_painel"] = "area"


def _ao_mudar_area(wkey: str, pkey: str) -> None:
    """Callback dos seletores de região/estado: reenquadra o mapa e mostra o ranking agregado."""
    _copiar(wkey, pkey)
    if pkey == "m2_regiao":  # trocar a região limpa o estado
        definir("m2_uf", "(Todos)")
    st.session_state.pop("m2_aviso_sem_historico", None)
    reg = st.session_state.get("m2_regiao", "(Brasil)")
    uf = st.session_state.get("m2_uf", "(Todos)")
    area_reg = None if reg == "(Brasil)" else reg
    area_uf = None if uf == "(Todos)" else uf
    st.session_state["m2_enquadramento"] = (area_reg, area_uf)
    st.session_state["m2_painel"] = "area" if (area_reg or area_uf) else "municipio"


def _ao_mudar_nivel() -> None:
    """Callback do seletor de nível: limpa seleção/área ou restaura o último nível se None."""
    novo = st.session_state.get("m2_nivel")
    if novo is None:
        st.session_state["m2_nivel"] = st.session_state.get("m2_nivel_ativo", config.NIVEIS_M2[0])
        return
    if novo != st.session_state.get("m2_nivel_ativo"):
        st.session_state["m2_nivel_ativo"] = novo
        st.session_state.pop("m2_codigo_ibge", None)
        st.session_state.pop("m2_aviso_sem_historico", None)
        definir("m2_regiao", "(Brasil)")
        definir("m2_uf", "(Todos)")
        st.session_state["m2_enquadramento"] = (None, None)
        st.session_state["m2_painel"] = "municipio"


def modo_o_que_produzir(refs: mu.Referencias) -> None:
    """Clique em município, estado ou região -> ranking dos produtos pelo índice relativo."""
    st.session_state.setdefault("m2_nivel_ativo", config.NIVEIS_M2[0])
    if st.session_state.get("m2_nivel") not in config.NIVEIS_M2:
        st.session_state["m2_nivel"] = st.session_state["m2_nivel_ativo"]
    nivel = st.session_state["m2_nivel"]
    st.session_state["m2_nivel_ativo"] = nivel

    st.subheader(":green[:material/location_on:] O que produzir?")
    st.write(_TEXTO_TOPO_M2[nivel])
    notas_fixas()
    st.session_state.setdefault("m2_painel", "municipio")
    st.session_state.setdefault("m2_enquadramento", (None, None))

    with st.expander("Controles", expanded=False):
        ajustes_t = controles_ajustes("m2")
        st.divider()
        st.markdown("**Área (opcional)** — aproxima o mapa e mostra o ranking agregado da região/estado")
        c1, c2 = st.columns(2)
        regioes = ["(Brasil)"] + sorted(refs.municipios["regiao"].unique())
        kw = persist("m2_regiao", "(Brasil)", regioes)
        kw["on_change"] = _ao_mudar_area
        regiao = c1.selectbox("Região", regioes, **kw)
        mun_reg = refs.municipios if regiao == "(Brasil)" else refs.municipios[refs.municipios["regiao"] == regiao]
        ufs = ["(Todos)"] + sorted(mun_reg["uf"].unique())
        nomes_uf = refs.municipios.drop_duplicates("uf").set_index("uf")["estado_nome"].to_dict()
        kw = persist("m2_uf", "(Todos)", ufs)
        kw["on_change"] = _ao_mudar_area
        uf = c2.selectbox("Estado", ufs, format_func=lambda u: u if u.startswith("(") else f"{u} — {nomes_uf.get(u, u)}", **kw)

    st.segmented_control("Nível do mapa", config.NIVEIS_M2, key="m2_nivel", on_change=_ao_mudar_nivel)

    area_reg = None if regiao == "(Brasil)" else regiao
    area_uf = None if uf == "(Todos)" else uf
    tem_area = bool(area_reg or area_uf)
    codigo = st.session_state.get("m2_codigo_ibge")
    enquadramento = st.session_state.get("m2_enquadramento", (None, None))

    if nivel == "Cidades":
        selecionado = codigo
    elif nivel == "Estados":
        selecionado = area_uf
    else:
        selecionado = area_reg if area_uf is None else None

    col_mapa, col_res = st.columns([3, 2])
    with col_mapa:
        fig, curvas = cache_figura_selecao(nivel, selecionado, enquadramento)
        st.session_state["m2_curvas"] = curvas
        st.plotly_chart(
            fig, width="stretch", key=_CHAVE_MAPA_M2[nivel], on_select=_ao_clicar_mapa, selection_mode="points",
            config={"scrollZoom": True, "displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
        )
        st.caption(
            "Use a roda do mouse para aproximar e arraste para mover. "
            "As cores identificam apenas as regiões (não representam previsão); o item selecionado recebe contorno escuro."
        )

    with col_res:
        aviso_sem_hist = st.session_state.get("m2_aviso_sem_historico")
        if aviso_sem_hist:
            st.warning(f"'{aviso_sem_hist}' está sem histórico de produção na base.", icon=":material/warning:")
        painel = st.session_state["m2_painel"]
        if painel == "area" and tem_area:
            _painel_area(refs, area_reg, area_uf, ajustes_t, codigo)
        elif codigo:
            _painel_municipio(refs, codigo, ajustes_t, tem_area)
        else:
            st.info(_MSG_ORIENTACAO_M2[nivel], icon=":material/ads_click:")

    with st.expander("O que é o índice relativo?"):
        st.markdown(texto_explicacao_indice())


def _painel_municipio(refs: mu.Referencias, codigo: str, ajustes_t: tuple, tem_area: bool) -> None:
    """Painel direito do Modo 2 para o município clicado."""
    if tem_area:
        st.button("Ver ranking agregado da área selecionada", key="m2_btn_area",
                  on_click=lambda: st.session_state.update(m2_painel="area"))
    if codigo not in refs.municipios.index:
        nomes = recurso_nomes_malha().set_index("codigo_ibge")["nome_exibicao"]
        st.markdown(f"### Município: {nomes.get(codigo, codigo)}")
        st.warning("Este município não tem histórico de produção na base; não há produtos para ranquear.")
        return
    m = refs.municipios.loc[codigo]
    st.markdown(f"### Município: {m['cidade']} — {m['uf']}")
    st.caption(f"Região {m['regiao']} · código IBGE {codigo}")
    df = cache_produtos((codigo,), ajustes_t)
    if df.empty:
        st.warning("Este município não tem nenhum produto com histórico na base.")
        return
    alerta_fora_faixa(mu.fora_da_faixa(df.drop_duplicates("codigo_ibge"), refs))
    st.plotly_chart(grafico_ranking(df, "Produtos ordenados pelo índice relativo"), width="stretch",
                    config={"displayModeBar": False}, key="m2_graf_mun")
    st.dataframe(tabela_ranking(df), hide_index=True, width="stretch", column_config=CFG_TABELA)
    st.caption("O rendimento bruto só é comparável dentro do mesmo produto; a ordem usa o índice relativo.")


def _painel_area(refs: mu.Referencias, regiao: str | None, uf: str | None, ajustes_t: tuple, codigo: str | None) -> None:
    """Painel direito do Modo 2 com o ranking agregado da região/estado."""
    mun = refs.municipios
    cods = mun.index[mun["uf"] == uf] if uf else mun.index[mun["regiao"] == regiao]
    nome_area = f"Estado: {mun.loc[mun['uf'] == uf, 'estado_nome'].iloc[0]} ({uf})" if uf else f"Região: {regiao}"
    if codigo and codigo in mun.index:
        st.button(f"Ver município selecionado ({mun.loc[codigo, 'nome_exibicao']})", key="m2_btn_mun",
                  on_click=lambda: st.session_state.update(m2_painel="municipio"))
    st.markdown(f"### Ranking agregado — {nome_area}")
    st.caption(
        f"Método: para cada produto, mediana simples do índice relativo (e do rendimento previsto) entre os "
        f"municípios da área que têm histórico do produto. Todos os municípios têm o mesmo peso, "
        f"independentemente da área cultivada. Municípios na área: {len(cods)}."
    )
    df = cache_produtos(tuple(sorted(cods)), ajustes_t)
    rk = mu.ranking_agregado(df)
    if rk.empty:
        st.warning("Nenhum produto com histórico nesta área.")
        return
    alerta_fora_faixa(mu.fora_da_faixa(df.drop_duplicates("codigo_ibge"), refs), len(cods))
    st.plotly_chart(grafico_ranking(rk, "Mediana do índice relativo por produto"), width="stretch",
                    config={"displayModeBar": False}, key="m2_graf_area")
    st.dataframe(tabela_ranking(rk, agregado=True), hide_index=True, width="stretch", column_config=CFG_TABELA)


# =============================================================================
# Modo 3 — Qual será o rendimento?
# =============================================================================
def _chave_clima(var: str, codigo: str) -> str:
    return f"m3_{var}_{codigo}"


def restaurar_clima(codigo: str) -> None:
    """Callback (fora do form): grava o clima base do município nos 6 campos."""
    refs = recurso_refs()
    for var in config.CLIMA:
        st.session_state[_chave_clima(var, codigo)] = float(refs.clima_base.loc[codigo, var])
    st.session_state.get("m3_clima_salvo", {}).pop(codigo, None)


def _ao_mudar_regiao_m3(wkey: str, pkey: str) -> None:
    _copiar(wkey, pkey)
    st.session_state.pop("m3_uf", None)
    st.session_state.pop("m3_codigo", None)


def _ao_mudar_uf_m3(wkey: str, pkey: str) -> None:
    _copiar(wkey, pkey)
    st.session_state.pop("m3_codigo", None)


def grafico_faixa(stats: pd.Series, previsto: float, unidade: str) -> go.Figure:
    """Faixa P10–P90 / P25–P75 e mediana históricas de UM produto, com a previsão marcada."""
    xmax = max(stats["p90"], previsto) * 1.1 + 1e-9
    xmin = min(stats["p10"], previsto) * 0.9
    fig = go.Figure()
    fig.add_shape(type="rect", x0=stats["p10"], x1=stats["p90"], y0=0.3, y1=0.7, fillcolor="#c7e9c0", line_width=0)
    fig.add_shape(type="rect", x0=stats["p25"], x1=stats["p75"], y0=0.3, y1=0.7, fillcolor="#74c476", line_width=0)
    fig.add_shape(type="line", x0=stats["mediana"], x1=stats["mediana"], y0=0.25, y1=0.75, line=dict(color="#00441b", width=3))
    fig.add_trace(go.Scatter(
        x=[previsto], y=[0.5], mode="markers+text",
        marker=dict(symbol="diamond", size=16, color="#1b5e20", line=dict(width=1.5, color="#0a2e0d")),
        text=["Previsão"], textposition="top center", hovertemplate=f"Previsão: %{{x:.3f}} {unidade}<extra></extra>",
        showlegend=False,
    ))
    for nome, x in (("P10", stats["p10"]), ("P25", stats["p25"]), ("Mediana", stats["mediana"]), ("P75", stats["p75"]), ("P90", stats["p90"])):
        fig.add_annotation(x=x, y=0.18, text=nome, showarrow=False, font=dict(size=10))
    fig.update_layout(
        title=dict(text="Distribuição histórica nacional deste produto", font=dict(size=14)),
        xaxis=dict(title=unidade, range=[xmin, xmax]), yaxis=dict(visible=False, range=[0, 1]),
        height=200, margin=dict(l=10, r=10, t=40, b=40),
    )
    return fig


def modo_rendimento(refs: mu.Referencias) -> None:
    """Variáveis informadas pelo usuário -> rendimento por hectare previsto."""
    st.subheader(":green[:material/calculate:] Qual será o rendimento?")
    st.write("Informe o local, o produto e as condições de clima para estimar o rendimento por hectare.")
    notas_fixas()
    mun = refs.municipios

    # --- Seleção de local e produto (fora do form: listas dependentes atualizam a cada escolha)
    c1, c2, c3 = st.columns(3)
    regioes = sorted(mun["regiao"].unique())
    kw = persist("m3_regiao", regioes[0], regioes)
    kw["on_change"] = _ao_mudar_regiao_m3
    regiao = c1.selectbox("Região", regioes, **kw)
    ufs = sorted(mun.loc[mun["regiao"] == regiao, "uf"].unique())
    nomes_uf = mun.drop_duplicates("uf").set_index("uf")["estado_nome"].to_dict()
    kw = persist("m3_uf", ufs[0], ufs)
    kw["on_change"] = _ao_mudar_uf_m3
    uf = c2.selectbox("Estado", ufs, format_func=lambda u: f"{u} — {nomes_uf.get(u, u)}", **kw)
    mun_uf = mun[mun["uf"] == uf].sort_values("cidade")
    cods = mun_uf.index.tolist()
    rotulos = mun_uf["cidade"].to_dict()
    codigo = c3.selectbox("Município", cods, format_func=lambda c: rotulos.get(c, c), **persist("m3_codigo", cods[0], cods))

    c4, c5, c6 = st.columns([1, 2, 1])
    tipos = ["Todos"] + sorted(refs.produtos["tipo_lavoura"].unique())
    tipo = c4.selectbox("Tipo de lavoura (filtra a lista)", tipos, **persist("m3_tipo", "Todos", tipos))
    validos = refs.produtos_validos(codigo)
    if tipo != "Todos":
        validos = [p for p in validos if refs.produtos.loc[p, "tipo_lavoura"] == tipo]
    if validos:
        produto = c5.selectbox("Produto (apenas com histórico no município)", validos,
                               **persist("m3_produto", validos[0], validos))
        c6.metric("Unidade", refs.produtos.loc[produto, "unidade"])
    else:
        produto = None
        c5.selectbox("Produto (apenas com histórico no município)", [], disabled=True, key="m3_produto_vazio")

    valido = bool(produto) and refs.combinacao_valida(codigo, produto)
    if not valido:
        st.error(mu.MSG_BLOQUEIO)

    # --- Clima: valores iniciais = clima base (ou o último valor calculado para este município)
    salvo = st.session_state.setdefault("m3_clima_salvo", {})
    for var in config.CLIMA:
        chave = _chave_clima(var, codigo)
        if chave not in st.session_state:
            st.session_state[chave] = float(salvo.get(codigo, {}).get(var, refs.clima_base.loc[codigo, var]))

    st.button("Restaurar clima base", icon=":material/restart_alt:", on_click=restaurar_clima, args=(codigo,), key="m3_btn_restaurar",
              help="Volta os 6 campos para a média do município nos últimos 10 anos da base (2015–2024).")
    if bool(refs.clima_base.loc[codigo].get("clima_base_fallback", False)):
        st.caption("Este município não tem dados em 2015–2024: o clima base usa a média de todos os anos disponíveis.")

    with st.form("m3_form"):
        st.markdown("**Variáveis de clima**")
        cols = st.columns(3)
        valores: dict[str, float] = {}
        for i, var in enumerate(config.CLIMA):
            passo, fmt = config.FORMATO_CLIMA[var]
            f = refs.faixas_clima[var]
            with cols[i % 3]:
                valores[var] = st.number_input(
                    config.ROTULOS_CLIMA[var], step=passo, format=fmt, key=_chave_clima(var, codigo),
                    help=f"Faixa observada na base: {f['min']:g} a {f['max']:g}",
                )
        st.caption(f"Ano de referência: **{config.ANO_REFERENCIA}** (parâmetro fixo, não editável).")
        calcular = st.form_submit_button("Calcular", type="primary", disabled=not valido)

    if calcular:
        salvo[codigo] = dict(valores)
        try:
            res = mu.prever_ponto(recurso_modelo(), refs, codigo, produto, valores)  # valida de novo no servidor
            st.session_state["m3_resultado"] = {"codigo": codigo, "produto": produto, "clima": dict(valores), **res}
        except mu.CombinacaoInvalidaError:
            st.session_state.pop("m3_resultado", None)
            st.error(mu.MSG_BLOQUEIO)
        except mu.CategoriaDesconhecidaError as e:
            st.error(f"Categoria desconhecida pelo modelo: {e}")
        except mu.MunicipioSemDadosError as e:
            st.error(str(e))
        except mu.EsquemaInvalidoError as e:
            st.error(f"Incompatibilidade de features: {e}")

    r = st.session_state.get("m3_resultado")
    if not r:
        return
    if r["codigo"] != codigo or r["produto"] != produto:
        st.info("Os resultados abaixo se referem a uma seleção anterior. Clique em **Calcular** para atualizar.")
    _mostrar_resultado_m3(refs, r)


def _mostrar_resultado_m3(refs: mu.Referencias, r: dict) -> None:
    """Saída do Modo 3."""
    m = refs.municipios.loc[r["codigo"]]
    unidade = refs.produtos.loc[r["produto"], "unidade"]
    stats = refs.referencia.stats(r["produto"])
    st.divider()
    st.markdown(f"### {r['produto']} em {m['cidade']} — {m['uf']}")
    c1, c2 = st.columns(2)
    c1.metric("Rendimento previsto por hectare", f"{r['rendimento_previsto']:.3f} {unidade}",
              help="Estimativa típica (próxima da mediana); tende a ficar um pouco abaixo da média real.")
    c2.metric("Índice relativo (0–100)", f"{r['indice']:.0f}",
              help="Percentil da previsão no histórico nacional deste produto.")
    st.caption(
        f"Índice {r['indice']:.0f}: a previsão supera cerca de {r['indice']:.0f}% dos rendimentos já "
        f"registrados para {r['produto']} no Brasil. {config.NOTA_ESTIMATIVA}"
    )
    alerta_fora_faixa(r["fora_faixa"])
    st.plotly_chart(grafico_faixa(stats, r["rendimento_previsto"], unidade), width="stretch",
                    config={"displayModeBar": False}, key="m3_graf_faixa")
    st.caption(
        f"Mediana histórica: {stats['mediana']:.3f} · P10–P90: {stats['p10']:.3f} a {stats['p90']:.3f} {unidade}. "
        "Esta faixa descreve o histórico nacional do produto; não é um intervalo de confiança da previsão."
    )
    resumo = {
        "Região": m["regiao"], "Estado": f"{m['uf']} — {m['estado_nome']}", "Município": m["cidade"],
        "Código IBGE": r["codigo"], "Produto": r["produto"], "Tipo de lavoura (derivado do produto)": r["tipo_lavoura"],
        "Ano de referência (fixo)": str(r["ano"]),
        **{config.ROTULOS_CLIMA[v]: f"{r['clima'][v]:.3f}" for v in config.CLIMA},
    }
    with st.expander("Valores de entrada usados", expanded=False):
        st.table(pd.DataFrame({"Variável": list(resumo), "Valor": list(resumo.values())}).set_index("Variável"))


# =============================================================================
# Estrutura geral
# =============================================================================
def barra_lateral(validacao: mu.ResultadoValidacao) -> None:
    """Sobre, limitações e status da verificação."""
    with st.sidebar:
        st.markdown("## :material/eco: AgroClima")
        st.write("Estimativa do **rendimento agrícola por hectare** nos municípios brasileiros, "
                 "a partir de um modelo XGBoost treinado com dados de 2000–2024.")
        st.markdown("### Limitações")
        st.markdown(
            "- Os resultados são **rendimento previsto por hectare**, e não volume de produção. A base não "
            "contém área plantada ou colhida, então o app não estima quanto o município produz no total.\n"
            f"- As estimativas assumem o nível tecnológico e as práticas do período mais recente da base "
            f"({config.ANO_REFERENCIA}). O modelo não projeta crescimento futuro.\n"
            "- Use os resultados para **comparar** municípios e produtos entre si; o valor absoluto previsto "
            "tem margem de erro relevante. É uma ferramenta de ranqueamento relativo, não de previsão precisa.\n"
            "- Na agregação por região ou estado (Modo 2), todos os municípios têm o mesmo peso (mediana "
            "simples), independentemente do tamanho da área cultivada.\n"
            "- A predição é uma estimativa do modelo, não garantia de produtividade.\n"
            f"- Unidade genérica “{config.UNIDADE}”: valores não são comparáveis entre produtos."
        )
        with st.expander("Verificação do ambiente"):
            st.write(":material/check_circle: Features, categorias, ano de referência e teste de fumaça OK.")
            rho = validacao.info.get("spearman_fumaca")
            if rho is not None:
                st.write(f"Spearman (amostra de {config.N_AMOSTRA_FUMACA} linhas da base): {rho:.3f}")
            st.caption("Esse teste só detecta erros de montagem das features; não mede a qualidade do modelo.")


def _ao_mudar_modo() -> None:
    """Guarda o modo ativo; se o controle for desmarcado (None), volta ao último modo."""
    novo = st.session_state.get("_w_modo")
    if novo is None:
        st.session_state["_w_modo"] = st.session_state.get("modo_ativo", config.MODOS[0])
    else:
        st.session_state["modo_ativo"] = novo


def main() -> None:
    """Ponto de entrada do app."""
    aplicar_estilo()
    st.title(":green[:material/eco:] AgroClima — rendimento agrícola estimado por município")
    try:
        refs = recurso_refs()
        recurso_modelo()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    validacao = recurso_validacao()
    if not validacao.ok:
        st.error("**A verificação de inicialização falhou. O app foi interrompido.**\n\n- " + "\n- ".join(validacao.erros))
        st.stop()
    for aviso in validacao.avisos:
        st.warning(aviso)
    barra_lateral(validacao)

    st.session_state.setdefault("modo_ativo", config.MODOS[0])
    st.session_state.setdefault("_w_modo", st.session_state["modo_ativo"])
    escolhido = st.segmented_control("Modo de uso", config.MODOS, key="_w_modo", on_change=_ao_mudar_modo)
    modo = escolhido or st.session_state["modo_ativo"]
    st.session_state["modo_ativo"] = modo

    # Apenas o modo ativo é executado (sem st.tabs).
    if modo == config.MODOS[0]:
        modo_onde_produzir(refs)
    elif modo == config.MODOS[1]:
        modo_o_que_produzir(refs)
    else:
        modo_rendimento(refs)


main()
