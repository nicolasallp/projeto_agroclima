"""Utilitários geográficos: normalização de nomes, malha municipal e mapas Plotly.

Os mapas usam ``plotly.graph_objects.Choroplethmap`` (MapLibre, sem token) com
``featureidkey="properties.codigo_ibge"``. O código IBGE (7 dígitos, texto) é a
única chave usada para ligar a malha às tabelas e para identificar cliques.
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import config

_RE_NAO_ALNUM = re.compile(r"[^0-9a-z\s]")
_RE_ESPACOS = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Normalização
# ---------------------------------------------------------------------------
def normalizar_nome(texto: Any) -> str:
    """Normaliza um nome para junção: minúsculas, sem acentos, sem pontuação, espaços aparados.

    Hífens e apóstrofos são removidos (``"D'Oeste" -> "doeste"``) e espaços
    repetidos viram um só.
    """
    if texto is None or (isinstance(texto, float) and np.isnan(texto)):
        return ""
    s = unicodedata.normalize("NFKD", str(texto))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    s = _RE_NAO_ALNUM.sub("", s.replace("-", " "))
    return _RE_ESPACOS.sub(" ", s).strip()


def normalizar_compacto(texto: Any) -> str:
    """Versão sem nenhum espaço (2ª tentativa de junção: ``"Embu Guaçu" == "Embu-Guaçu"``)."""
    return normalizar_nome(texto).replace(" ", "")


def codigo_7(valor: Any) -> str:
    """Converte um código IBGE qualquer em texto de 7 dígitos."""
    return str(int(float(valor))).zfill(7)


# ---------------------------------------------------------------------------
# Malha
# ---------------------------------------------------------------------------
def carregar_malha(caminho=config.ARQ_MALHA) -> dict:
    """Lê o GeoJSON simplificado gerado pelo ``build_reference.py``."""
    with open(caminho, encoding="utf-8") as f:
        return json.load(f)


def subconjunto_malha(malha: dict, codigos: Iterable[str]) -> dict:
    """Retorna um FeatureCollection apenas com os municípios pedidos (sem copiar geometrias)."""
    alvo = set(codigos)
    feats = [ft for ft in malha["features"] if ft["properties"]["codigo_ibge"] in alvo]
    return {"type": "FeatureCollection", "features": feats}


def _coords(geom: dict) -> np.ndarray:
    """Achata as coordenadas de um Polygon/MultiPolygon em um array (n, 2)."""
    if geom["type"] == "Polygon":
        aneis = geom["coordinates"]
    else:  # MultiPolygon
        aneis = [anel for poly in geom["coordinates"] for anel in poly]
    return np.concatenate([np.asarray(a, dtype=float) for a in aneis])


def bbox_por_municipio(malha: dict) -> pd.DataFrame:
    """Caixa envolvente (lon/lat mín./máx.) de cada município da malha."""
    linhas = []
    for ft in malha["features"]:
        c = _coords(ft["geometry"])
        linhas.append(
            (ft["properties"]["codigo_ibge"], c[:, 0].min(), c[:, 0].max(), c[:, 1].min(), c[:, 1].max())
        )
    return pd.DataFrame(linhas, columns=["codigo_ibge", "lon_min", "lon_max", "lat_min", "lat_max"])


def centro_zoom(bbox: pd.DataFrame) -> tuple[dict[str, float], float]:
    """Calcula centro e zoom aproximados para enquadrar um conjunto de caixas."""
    if bbox.empty:
        return dict(config.CENTRO_BRASIL), config.ZOOM_BRASIL
    lon0, lon1 = bbox["lon_min"].min(), bbox["lon_max"].max()
    lat0, lat1 = bbox["lat_min"].min(), bbox["lat_max"].max()
    extensao = max(lon1 - lon0, (lat1 - lat0) * 1.3, 0.05)
    zoom = float(np.clip(np.log2(360.0 / extensao) - 0.9, 2.5, 11.0))
    return {"lat": float((lat0 + lat1) / 2), "lon": float((lon0 + lon1) / 2)}, zoom


# ---------------------------------------------------------------------------
# Seleção por clique
# ---------------------------------------------------------------------------
def codigo_do_ponto(ponto: dict, codigos_por_curva: Sequence[Sequence[str]] | None = None) -> str | None:
    """Extrai o ``codigo_ibge`` de um ponto selecionado no ``st.plotly_chart``.

    Ordem de preferência: ``customdata`` (sempre gravamos o código lá),
    ``location`` e, por último, ``point_index`` na lista de códigos da curva.
    Nunca usa o nome do município.
    """
    cd = ponto.get("customdata")
    if cd is not None:
        valor = cd[0] if isinstance(cd, (list, tuple)) else cd
        if valor is not None and str(valor).strip():
            return str(valor)
    loc = ponto.get("location")
    if loc:
        return str(loc)
    if codigos_por_curva is not None:
        curva = ponto.get("curve_number")
        idx = ponto.get("point_index", ponto.get("point_number"))
        if curva is not None and idx is not None and curva < len(codigos_por_curva):
            lista = codigos_por_curva[curva]
            if 0 <= idx < len(lista):
                return str(lista[idx])
    return None


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def _layout(fig: go.Figure, centro: dict, zoom: float, uirevision: str, altura: int) -> go.Figure:
    fig.update_layout(
        map=dict(style=config.ESTILO_MAPA, center=centro, zoom=zoom),
        margin=dict(l=0, r=0, t=0, b=0),
        height=altura,
        uirevision=uirevision,
        legend=dict(yanchor="bottom", y=0.01, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.8)"),
        clickmode="event+select",
    )
    return fig


def figura_mapa_predicao(
    malha: dict,
    df: pd.DataFrame,
    coluna_cor: str,
    titulo_cor: str,
    codigos_sem_historico: Sequence[str],
    hover_cidades: dict[str, str],
    altura: int = 620,
) -> go.Figure:
    """Mapa coroplético do Modo 1.

    ``df`` deve ter ``codigo_ibge``, ``nome_exibicao``, ``rendimento_previsto``,
    ``indice`` e a coluna de cor. Municípios sem histórico ficam em cinza numa
    camada separada (cada feição aparece uma única vez, para manter o mapa leve).
    """
    fig = go.Figure()
    sem = list(codigos_sem_historico)
    if sem:
        fig.add_trace(
            go.Choroplethmap(
                geojson=subconjunto_malha(malha, sem),
                featureidkey="properties.codigo_ibge",
                locations=sem,
                z=np.zeros(len(sem)),
                colorscale=[[0, "#d9d9d9"], [1, "#d9d9d9"]],
                showscale=False,
                marker=dict(line=dict(width=0.2, color="#ffffff")),
                customdata=np.array([[c, hover_cidades.get(c, c)] for c in sem], dtype=object),
                hovertemplate="%{customdata[1]}<br>Sem histórico de produção<extra></extra>",
                name="Sem histórico de produção",
                showlegend=True,
            )
        )
    if not df.empty:
        is_indice = coluna_cor == "indice"
        fig.add_trace(
            go.Choroplethmap(
                geojson=subconjunto_malha(malha, df["codigo_ibge"]),
                featureidkey="properties.codigo_ibge",
                locations=df["codigo_ibge"].tolist(),
                z=df[coluna_cor].to_numpy(dtype=float),
                zmin=0 if is_indice else float(df[coluna_cor].quantile(0.02)),
                zmax=100 if is_indice else float(df[coluna_cor].quantile(0.98)),
                colorscale="YlGn",
                marker=dict(line=dict(width=0.2, color="#ffffff")),
                colorbar=dict(title=dict(text=titulo_cor), thickness=14, len=0.75),
                customdata=np.stack(
                    [
                        df["codigo_ibge"].to_numpy(dtype=object),
                        df["nome_exibicao"].to_numpy(dtype=object),
                        df["rendimento_previsto"].round(3).to_numpy(dtype=object),
                        df["indice"].round(1).to_numpy(dtype=object),
                    ],
                    axis=1,
                ),
                hovertemplate=(
                    "<b>%{customdata[1]}</b><br>"
                    f"Rendimento previsto: %{{customdata[2]}} {config.UNIDADE}<br>"
                    "Índice relativo: %{customdata[3]}<extra></extra>"
                ),
                name="Previsão",
                showlegend=False,
            )
        )
    return _layout(fig, dict(config.CENTRO_BRASIL), config.ZOOM_BRASIL, "mapa-modo1", altura)


def figura_mapa_selecao(
    malha: dict,
    municipios: pd.DataFrame,
    codigo_selecionado: str | None,
    centro: dict,
    zoom: float,
    uirevision: str,
    altura: int = 620,
) -> tuple[go.Figure, list[list[str]]]:
    """Mapa neutro do Modo 2, com o município selecionado destacado.

    Retorna a figura e a lista de códigos de cada curva (para traduzir
    ``point_index`` em ``codigo_ibge`` caso o ``customdata`` não venha no evento).
    ``municipios`` deve ter ``codigo_ibge`` e ``nome_exibicao`` para TODAS as feições da malha.
    """
    codigos = [ft["properties"]["codigo_ibge"] for ft in malha["features"]]
    nomes = municipios.set_index("codigo_ibge")["nome_exibicao"].to_dict()
    base = [c for c in codigos if c != codigo_selecionado]
    fig = go.Figure()
    fig.add_trace(
        go.Choroplethmap(
            geojson=subconjunto_malha(malha, base) if codigo_selecionado else malha,
            featureidkey="properties.codigo_ibge",
            locations=base,
            z=np.zeros(len(base)),
            colorscale=[[0, "#cfe3c8"], [1, "#cfe3c8"]],
            showscale=False,
            marker=dict(line=dict(width=0.3, color="#7f9c76"), opacity=0.85),
            selected=dict(marker=dict(opacity=0.85)),
            unselected=dict(marker=dict(opacity=0.85)),
            customdata=np.array([[c, nomes.get(c, c)] for c in base], dtype=object),
            hovertemplate="%{customdata[1]}<extra></extra>",
            name="Municípios",
            showlegend=False,
        )
    )
    curvas = [base]
    if codigo_selecionado:
        fig.add_trace(
            go.Choroplethmap(
                geojson=subconjunto_malha(malha, [codigo_selecionado]),
                featureidkey="properties.codigo_ibge",
                locations=[codigo_selecionado],
                z=[1.0],
                colorscale=[[0, "#1b5e20"], [1, "#1b5e20"]],
                showscale=False,
                marker=dict(line=dict(width=2.5, color="#0a2e0d"), opacity=1.0),
                selected=dict(marker=dict(opacity=1.0)),
                unselected=dict(marker=dict(opacity=1.0)),
                customdata=np.array([[codigo_selecionado, nomes.get(codigo_selecionado, "")]], dtype=object),
                hovertemplate="<b>%{customdata[1]}</b> (selecionado)<extra></extra>",
                name="Selecionado",
                showlegend=False,
            )
        )
        curvas.append([codigo_selecionado])
    return _layout(fig, centro, zoom, uirevision, altura), curvas
