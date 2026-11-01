"""Gera as tabelas auxiliares do app a partir da base única e dos dados do IBGE.

Uso:
    python build_reference.py

Regras:
- A base ``agroclima_anual.csv`` é usada EXATAMENTE como está: nenhum filtro,
  remoção de linhas ou tratamento de outliers.
- ``codigo_ibge``, ``uf`` e a geometria vêm do IBGE (baixados uma vez para ``data/ibge/``).
- ``referencia_produto`` é calculada em escala ORIGINAL (sem log).
- ``clima_base`` usa apenas os últimos ``ANOS_CLIMA_BASE`` anos da base.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import config
from geo_utils import codigo_7, normalizar_compacto, normalizar_nome

LINHAS_RELATORIO: list[str] = []


def log(msg: str = "") -> None:
    """Imprime e guarda a linha no relatório."""
    print(msg, flush=True)
    LINHAS_RELATORIO.append(msg)


def falhar(msg: str) -> None:
    """Interrompe o script com mensagem clara (sem prosseguir com dados parciais)."""
    print("\n[ERRO] " + msg, file=sys.stderr)
    sys.exit(1)


MSG_MANUAL = (
    "Não foi possível obter os dados do IBGE. Coloque manualmente em 'data/ibge/':\n"
    f"  - {config.ARQ_IBGE_MUNICIPIOS.name}: resposta JSON de {config.URL_IBGE_MUNICIPIOS}\n"
    f"  - {config.ARQ_IBGE_MALHA.name}: GeoJSON de "
    f"{config.URL_IBGE_MALHA_TEMPLATE.format(qualidade='minima')}\n"
    "e rode o script novamente."
)


# ---------------------------------------------------------------------------
# Download e validação IBGE
# ---------------------------------------------------------------------------
def _validar_municipios_ibge(dados) -> None:
    if not isinstance(dados, list) or len(dados) < 5000:
        raise ValueError(f"lista de municípios inesperada ({type(dados).__name__}, {len(dados) if hasattr(dados, '__len__') else '?'} itens)")
    amostra = dados[0]
    for campo in ("id", "nome"):
        if campo not in amostra:
            raise ValueError(f"campo '{campo}' ausente na resposta de municípios")


def _validar_malha(geo) -> None:
    if not isinstance(geo, dict) or geo.get("type") != "FeatureCollection":
        raise ValueError("a malha não é um FeatureCollection")
    feats = geo.get("features", [])
    if abs(len(feats) - config.MALHA_N_ESPERADO) > 50:
        raise ValueError(f"a malha tem {len(feats)} feições (esperado ~{config.MALHA_N_ESPERADO})")
    if "codarea" not in feats[0].get("properties", {}) and "codigo_ibge" not in feats[0].get("properties", {}):
        raise ValueError(f"campo do código não encontrado nas propriedades: {feats[0].get('properties')}")


def baixar_ibge() -> tuple[list, dict]:
    """Baixa (ou reutiliza) municípios e malha do IBGE, validando a estrutura."""
    config.DIR_IBGE.mkdir(parents=True, exist_ok=True)

    if not config.ARQ_IBGE_MUNICIPIOS.exists():
        log(f"Baixando municípios do IBGE: {config.URL_IBGE_MUNICIPIOS}")
        try:
            r = requests.get(config.URL_IBGE_MUNICIPIOS, timeout=120)
            r.raise_for_status()
            dados = r.json()
            _validar_municipios_ibge(dados)
        except Exception as e:  # noqa: BLE001
            falhar(f"Falha ao baixar municípios ({e}).\n{MSG_MANUAL}")
        config.ARQ_IBGE_MUNICIPIOS.write_text(json.dumps(dados, ensure_ascii=False), encoding="utf-8")
    else:
        log(f"Reutilizando {config.ARQ_IBGE_MUNICIPIOS}")

    if not config.ARQ_IBGE_MALHA.exists():
        geo, erros = None, []
        for q in config.QUALIDADES_MALHA:
            url = config.URL_IBGE_MALHA_TEMPLATE.format(qualidade=q)
            log(f"Baixando malha do IBGE (qualidade={q}): {url}")
            try:
                r = requests.get(url, timeout=300)
                r.raise_for_status()
                geo = r.json()
                _validar_malha(geo)
                break
            except Exception as e:  # noqa: BLE001
                erros.append(f"qualidade={q}: {e}")
                geo = None
        if geo is None:
            falhar(
                "Falha ao baixar a malha ("
                + "; ".join(erros)
                + "). Consulte também https://servicodados.ibge.gov.br/api/docs/malhas?versao=4\n"
                + MSG_MANUAL
            )
        config.ARQ_IBGE_MALHA.write_text(json.dumps(geo, ensure_ascii=False), encoding="utf-8")
    else:
        log(f"Reutilizando {config.ARQ_IBGE_MALHA}")

    try:
        dados = json.loads(config.ARQ_IBGE_MUNICIPIOS.read_text(encoding="utf-8"))
        _validar_municipios_ibge(dados)
        geo = json.loads(config.ARQ_IBGE_MALHA.read_text(encoding="utf-8"))
        _validar_malha(geo)
    except Exception as e:  # noqa: BLE001
        falhar(f"Arquivo do IBGE inválido em data/ibge/ ({e}). Apague-o e rode novamente.\n{MSG_MANUAL}")
    return dados, geo


def tabela_ibge(dados: list) -> pd.DataFrame:
    """Achata a resposta de municípios do IBGE (UF e região pela hierarquia real)."""
    linhas = []
    sem_micro = 0
    for m in dados:
        uf = None
        micro = m.get("microrregiao")
        if micro and micro.get("mesorregiao"):
            uf = micro["mesorregiao"].get("UF")
        if uf is None:
            sem_micro += 1
            imediata = m.get("regiao-imediata") or {}
            uf = (imediata.get("regiao-intermediaria") or {}).get("UF")
        if uf is None:
            falhar(f"Município IBGE {m.get('id')} sem UF identificável na resposta.")
        linhas.append(
            {
                "codigo_ibge": codigo_7(m["id"]),
                "nome_ibge": m["nome"],
                "uf": uf["sigla"],
                "estado_nome": uf["nome"],
                "regiao_ibge": uf["regiao"]["nome"],
            }
        )
    df = pd.DataFrame(linhas)
    log(f"Municípios IBGE: {len(df)} (sem microrregião, UF via região imediata: {sem_micro})")
    if df["codigo_ibge"].duplicated().any():
        falhar("Códigos IBGE duplicados na resposta de municípios.")
    return df


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------
def carregar_base() -> pd.DataFrame:
    """Lê a base única sem qualquer filtro."""
    if not config.ARQ_BASE.exists():
        falhar(f"Base não encontrada: {config.ARQ_BASE}")
    df = pd.read_csv(config.ARQ_BASE, encoding="utf-8")
    faltando = set(config.FEATURES + [config.ALVO]) - set(df.columns)
    if faltando:
        falhar(f"Colunas ausentes na base: {sorted(faltando)}")
    return df


def validar_base(df: pd.DataFrame) -> None:
    """Relata problemas da base SEM alterá-la."""
    y = df[config.ALVO]
    log("\n== Validação da base (nenhuma linha foi alterada) ==")
    log(f"Linhas: {len(df):,}".replace(",", "."))
    log(f"rendimento_ha: nulos={int(y.isna().sum())}, infinitos={int(np.isinf(y).sum())}, <=0={int((y <= 0).sum())}")
    nulos = df[config.FEATURES].isna().sum()
    log("Nulos nas features: " + (", ".join(f"{c}={n}" for c, n in nulos.items() if n) or "nenhum"))
    for c in config.CLIMA:
        n_inf = int(np.isinf(df[c]).sum())
        if n_inf:
            log(f"Infinitos em {c}: {n_inf}")


# ---------------------------------------------------------------------------
# Tabelas
# ---------------------------------------------------------------------------
def montar_municipios(base: pd.DataFrame, ibge: pd.DataFrame) -> pd.DataFrame:
    """Associa cada (cidade, estado) da base a um ``codigo_ibge``."""
    pares = base[["cidade", "estado", "regiao"]].drop_duplicates()
    regioes_por_par = pares.groupby(["cidade", "estado"])["regiao"].nunique()
    if (regioes_por_par > 1).any():
        falhar(f"Pares (cidade, estado) com mais de uma região: {regioes_por_par[regioes_por_par > 1].index.tolist()[:10]}")
    pares = pares.reset_index(drop=True)
    pares["codigo_ibge"] = pd.NA
    pares["origem_codigo"] = pd.NA

    # 1) de-para manual (opcional)
    if config.ARQ_DE_PARA.exists():
        dp = pd.read_csv(config.ARQ_DE_PARA, dtype=str, encoding="utf-8").dropna(how="all")
        if not dp.empty:
            dp["codigo_ibge"] = dp["codigo_ibge"].map(codigo_7)
            mapa = {(r.cidade, r.estado): r.codigo_ibge for r in dp.itertuples()}
            for i, r in pares.iterrows():
                cod = mapa.get((r["cidade"], r["estado"]))
                if cod:
                    pares.at[i, "codigo_ibge"] = cod
                    pares.at[i, "origem_codigo"] = "de_para"
            log(f"De-para manual aplicado: {int((pares['origem_codigo'] == 'de_para').sum())} municípios")

    # 2) nome normalizado + UF (o estado da base é a sigla) e 3) nome compacto + UF
    estado_e_sigla = base["estado"].str.len().max() == 2
    chave_estado = "uf" if estado_e_sigla else "estado_nome"
    ibge = ibge.copy()
    ibge["k_estado"] = ibge[chave_estado].map(normalizar_nome)
    pares["k_estado"] = pares["estado"].map(normalizar_nome)
    for etapa, fn in (("nome_normalizado", normalizar_nome), ("nome_compacto", normalizar_compacto)):
        ibge["k_cidade"] = ibge["nome_ibge"].map(fn)
        dup = ibge.duplicated(["k_cidade", "k_estado"], keep=False)
        lookup = ibge.loc[~dup].set_index(["k_cidade", "k_estado"])["codigo_ibge"].to_dict()
        pend = pares["codigo_ibge"].isna()
        cods = [lookup.get((fn(c), e)) for c, e in zip(pares.loc[pend, "cidade"], pares.loc[pend, "k_estado"])]
        pares.loc[pend, "codigo_ibge"] = cods
        pares.loc[pend & pares["codigo_ibge"].notna(), "origem_codigo"] = etapa

    casados = pares[pares["codigo_ibge"].notna()].copy()
    nao_casados = pares[pares["codigo_ibge"].isna()]

    dup_cod = casados[casados["codigo_ibge"].duplicated(keep=False)]
    if not dup_cod.empty:
        falhar(f"Mais de um município da base associado ao mesmo código IBGE:\n{dup_cod.to_string()}")

    casados = casados.merge(ibge[["codigo_ibge", "uf", "estado_nome", "regiao_ibge", "nome_ibge"]], on="codigo_ibge", how="left")
    div_regiao = casados[casados["regiao"] != casados["regiao_ibge"]]
    log("\n== Junção base x IBGE ==")
    log(f"Municípios na base (cidade, estado): {len(pares)}")
    log(f"Casados: {len(casados)} " + str(casados["origem_codigo"].value_counts().to_dict()))
    log(f"Da base sem correspondência no IBGE: {len(nao_casados)}")
    for r in nao_casados.itertuples():
        log(f"   - {r.cidade} / {r.estado}")
    if not div_regiao.empty:
        log(f"Atenção: {len(div_regiao)} municípios com região da base diferente da do IBGE (mantida a da base).")

    casados["nome_exibicao"] = casados["cidade"] + " - " + casados["uf"]
    cols = ["codigo_ibge", "regiao", "estado", "cidade", "uf", "estado_nome", "nome_exibicao"]
    return casados[cols].sort_values(["regiao", "estado", "cidade"]).reset_index(drop=True)


def montar_produtos(base: pd.DataFrame) -> pd.DataFrame:
    """Tabela produto -> tipo_lavoura (grafia da base), com validação de unicidade."""
    tipos = base.groupby("produto")["tipo_lavoura"].unique()
    ambiguos = tipos[tipos.map(len) > 1]
    if not ambiguos.empty:
        falhar(f"Produtos com mais de um tipo_lavoura: {ambiguos.to_dict()}")
    prod = pd.DataFrame({"produto": tipos.index, "tipo_lavoura": tipos.map(lambda v: v[0]).values})
    prod["unidade"] = config.UNIDADE
    return prod.sort_values("produto").reset_index(drop=True)


def montar_clima_base(base: pd.DataFrame, municipios: pd.DataFrame) -> pd.DataFrame:
    """Média das 6 variáveis por município nos últimos ``ANOS_CLIMA_BASE`` anos (com fallback)."""
    ano_max = int(base["ano"].max())
    ano_ini = ano_max - config.ANOS_CLIMA_BASE + 1
    # Clima é do município-ano (igual entre produtos): deduplica para não pesar anos com mais produtos.
    anual = base.groupby(["cidade", "estado", "ano"], as_index=False)[config.CLIMA].mean()
    janela = anual[anual["ano"].between(ano_ini, ano_max)]
    media_janela = janela.groupby(["cidade", "estado"])[config.CLIMA].mean()
    media_total = anual.groupby(["cidade", "estado"])[config.CLIMA].mean()
    clima = media_total.copy()
    clima["clima_base_fallback"] = True
    clima.loc[media_janela.index, config.CLIMA] = media_janela
    clima.loc[media_janela.index, "clima_base_fallback"] = False
    clima = clima.reset_index().merge(municipios[["codigo_ibge", "cidade", "estado"]], on=["cidade", "estado"], how="inner")
    log("\n== Clima base ==")
    log(f"Janela: {ano_ini}-{ano_max} ({config.ANOS_CLIMA_BASE} anos)")
    log(f"Municípios com clima_base_fallback (sem dados na janela): {int(clima['clima_base_fallback'].sum())}")
    return clima[["codigo_ibge"] + config.CLIMA + ["clima_base_fallback"]]


def montar_referencia(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Distribuição de ``rendimento_ha`` em ESCALA ORIGINAL por produto (todos os anos)."""
    v = base[["produto", config.ALVO]].copy()
    v = v[v[config.ALVO].notna() & np.isfinite(v[config.ALVO])]  # só ignora nulos no cálculo
    g = v.groupby("produto")[config.ALVO]
    stats = pd.DataFrame(
        {
            "n": g.size(),
            "media": g.mean(),
            "mediana": g.median(),
            "p10": g.quantile(0.10),
            "p25": g.quantile(0.25),
            "p75": g.quantile(0.75),
            "p90": g.quantile(0.90),
            "minimo": g.min(),
            "maximo": g.max(),
        }
    ).reset_index()
    stats["unidade"] = config.UNIDADE
    valores = v.sort_values(["produto", config.ALVO]).reset_index(drop=True)
    return stats, valores


def processar_malha(geo: dict) -> dict:
    """Renomeia ``codarea`` -> ``codigo_ibge`` (7 dígitos) e simplifica se a malha for pesada."""
    feats = []
    for ft in geo["features"]:
        props = ft.get("properties", {})
        cod = codigo_7(props.get("codigo_ibge", props.get("codarea")))
        feats.append({"type": "Feature", "properties": {"codigo_ibge": cod}, "geometry": ft["geometry"]})
    saida = {"type": "FeatureCollection", "features": feats}
    tamanho_mb = len(json.dumps(saida)) / 1e6
    if tamanho_mb > config.MALHA_TAMANHO_MAX_MB:
        from shapely.geometry import mapping, shape

        log(f"Malha com {tamanho_mb:.1f} MB: simplificando (tolerância {config.MALHA_TOLERANCIA_SIMPLIFICACAO}).")
        for ft in feats:
            g = shape(ft["geometry"]).simplify(config.MALHA_TOLERANCIA_SIMPLIFICACAO, preserve_topology=True)
            ft["geometry"] = json.loads(json.dumps(mapping(g)))
        tamanho_mb = len(json.dumps(saida)) / 1e6
    log(f"Malha final: {len(feats)} feições, {tamanho_mb:.1f} MB")
    return saida


# ---------------------------------------------------------------------------
# Estados e regiões (dissolução da malha municipal com shapely)
# ---------------------------------------------------------------------------
def _poligonos(g) -> list:
    """Extrai os polígonos de qualquer geometria (make_valid pode devolver coleções)."""
    from shapely.geometry import MultiPolygon, Polygon

    if isinstance(g, Polygon):
        return [g] if not g.is_empty else []
    if isinstance(g, MultiPolygon):
        return [p for p in g.geoms if not p.is_empty]
    if hasattr(g, "geoms"):
        return [p for sub in g.geoms for p in _poligonos(sub)]
    return []


def _limpar_geometria(g, simplificar: bool):
    """make_valid/buffer(0), fecha frestas, remove furos minúsculos e (se pedido) simplifica."""
    from shapely import make_valid, set_precision
    from shapely.geometry import MultiPolygon, Polygon

    eps = config.MALHA_BUFFER_FECHAMENTO
    g = make_valid(g).buffer(0)
    g = g.buffer(eps, join_style="mitre").buffer(-eps, join_style="mitre")
    polys = [
        Polygon(p.exterior, [i for i in p.interiors if Polygon(i).area >= config.MALHA_AREA_MIN_FURO])
        for p in _poligonos(g)
    ]
    g = MultiPolygon(polys) if len(polys) > 1 else polys[0]
    if simplificar:
        g = g.simplify(config.MALHA_TOLERANCIA_SIMPLIFICACAO, preserve_topology=True)
    g = make_valid(set_precision(g, 1e-4))
    polys = _poligonos(g)
    return MultiPolygon(polys) if len(polys) > 1 else polys[0]


def _feature(geom, props: dict) -> dict:
    from shapely.geometry import mapping

    return {"type": "Feature", "properties": props, "geometry": json.loads(json.dumps(mapping(geom)))}


def _tamanho_mb(fc: dict) -> float:
    return len(json.dumps(fc, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) / 1e6


def regiao_por_uf(municipios: pd.DataFrame, ibge: pd.DataFrame) -> dict[str, str]:
    """Região de cada UF: grafia da base; ``regiao_ibge`` apenas para UFs ausentes da base."""
    por_uf = municipios.groupby("uf")["regiao"].unique()
    ambiguas = por_uf[por_uf.map(len) > 1]
    if not ambiguas.empty:
        falhar(f"UFs com mais de uma região na base: {ambiguas.to_dict()}")
    mapa = {uf: v[0] for uf, v in por_uf.items()}
    for uf, reg in ibge.drop_duplicates("uf").set_index("uf")["regiao_ibge"].items():
        mapa.setdefault(uf, reg)
    return mapa


def montar_estados_regioes(malha: dict, ibge: pd.DataFrame, reg_uf: dict[str, str]) -> tuple[dict, dict]:
    """Dissolve municípios -> 27 estados -> 5 regiões, valida e registra no relatório."""
    from shapely import unary_union
    from shapely.geometry import shape

    uf_por_cod = ibge.set_index("codigo_ibge")["uf"].to_dict()
    nome_uf = ibge.drop_duplicates("uf").set_index("uf")["estado_nome"].to_dict()
    geoms_uf: dict[str, list] = {}
    sem_uf = []
    for ft in malha["features"]:
        cod = ft["properties"]["codigo_ibge"]
        uf = uf_por_cod.get(cod)
        if uf is None:
            sem_uf.append(cod)
            continue
        geoms_uf.setdefault(uf, []).append(shape(ft["geometry"]))
    if sem_uf:
        falhar(f"Feições da malha sem UF na tabela do IBGE: {sem_uf[:10]}")

    def _gerar(simplificar: bool) -> tuple[dict, dict]:
        estados_geom = {uf: _limpar_geometria(unary_union(gs), simplificar) for uf, gs in sorted(geoms_uf.items())}
        estados = {
            "type": "FeatureCollection",
            "features": [
                _feature(g, {"uf": uf, "estado_nome": nome_uf[uf], "regiao": reg_uf[uf]})
                for uf, g in estados_geom.items()
            ],
        }
        por_regiao: dict[str, list] = {}
        for uf, g in estados_geom.items():
            por_regiao.setdefault(reg_uf[uf], []).append(g)
        regioes = {
            "type": "FeatureCollection",
            "features": [
                _feature(_limpar_geometria(unary_union(gs), simplificar), {"regiao": reg})
                for reg, gs in sorted(por_regiao.items())
            ],
        }
        return estados, regioes

    estados, regioes = _gerar(simplificar=False)
    if _tamanho_mb(estados) + _tamanho_mb(regioes) > config.MALHA_AGREGADA_MAX_MB:
        log(f"Estados/regiões acima de {config.MALHA_AGREGADA_MAX_MB} MB: simplificando "
            f"(tolerância {config.MALHA_TOLERANCIA_SIMPLIFICACAO}).")
        estados, regioes = _gerar(simplificar=True)

    # Validação
    log("\n== Estados e regiões (dissolvidos da malha municipal) ==")
    ufs = [f["properties"]["uf"] for f in estados["features"]]
    regs = [f["properties"]["regiao"] for f in regioes["features"]]
    ufs_ibge = set(ibge["uf"])
    if len(ufs) != 27 or set(ufs) != ufs_ibge:
        falhar(f"Esperados 27 estados; obtidos {len(ufs)}. Diferença com o IBGE: {sorted(set(ufs) ^ ufs_ibge)}")
    if len(regs) != 5:
        falhar(f"Esperadas 5 regiões; obtidas {len(regs)}: {regs}")
    sem_cor = [r for r in regs if normalizar_nome(r) not in config.CORES_REGIOES]
    if sem_cor:
        falhar(f"Regiões sem cor em config.CORES_REGIOES: {sem_cor}")
    regioes_base = set(reg_uf.values())
    if set(regs) != regioes_base:
        falhar(f"Regiões sem correspondência: {sorted(set(regs) ^ regioes_base)}")
    for fc, nome in ((estados, "estado"), (regioes, "região")):
        for f in fc["features"]:
            g = shape(f["geometry"])
            if g.is_empty or not g.is_valid:
                falhar(f"Geometria inválida/vazia para {nome} {f['properties']}")
    log(f"Estados: {len(ufs)} (todas as UFs do IBGE com geometria e região) | {_tamanho_mb(estados):.2f} MB")
    log(f"Regiões: {len(regs)} {regs} | {_tamanho_mb(regioes):.2f} MB")
    return estados, regioes


def comparar_categorias_modelo(categorias: dict[str, list[str]]) -> None:
    """Se o modelo gravar categorias (xgboost >= 3.1), compara com as da base e salva .ubj.gz."""
    try:
        import gzip
        import xgboost as xgb

        booster = xgb.Booster()
        booster.load_model(str(config.ARQ_MODELO))
        if not config.ARQ_MODELO_UBJ.exists():
            raw_ubj = booster.save_raw(raw_format="ubj")
            with gzip.open(config.ARQ_MODELO_UBJ, "wb", compresslevel=6) as f_gz:
                f_gz.write(raw_ubj)
            log(f"Modelo compacto gravado em {config.ARQ_MODELO_UBJ.name}")
        cats = dict(booster.get_categories(export_to_arrow=True).to_arrow())
    except Exception as e:  # noqa: BLE001
        log(f"(Não foi possível ler categorias gravadas no modelo: {e})")
        return
    for col, lista in categorias.items():
        arr = cats.get(col)
        if arr is None:
            log(f"Modelo sem categorias gravadas para '{col}'.")
            continue
        iguais = arr.to_pylist() == lista
        log(f"Categorias de '{col}': base={len(lista)} modelo={len(arr)} -> {'IGUAIS' if iguais else 'DIFERENTES'}")


# ---------------------------------------------------------------------------
def main() -> None:
    """Executa todo o pipeline de referência e imprime o relatório."""
    config.DIR_REF.mkdir(parents=True, exist_ok=True)
    dados_ibge, geo = baixar_ibge()
    ibge = tabela_ibge(dados_ibge)

    log("\nLendo a base...")
    base = carregar_base()
    validar_base(base)
    log(f"Produtos: {base['produto'].nunique()} | Municípios (cidade, estado): "
        f"{base[['cidade', 'estado']].drop_duplicates().shape[0]} | Anos: {base['ano'].min()}-{base['ano'].max()}")

    municipios = montar_municipios(base, ibge)
    malha = processar_malha(geo)
    cod_malha = {ft["properties"]["codigo_ibge"] for ft in malha["features"]}
    sem_malha = set(municipios["codigo_ibge"]) - cod_malha
    malha_sem_base = cod_malha - set(municipios["codigo_ibge"])
    log(f"Municípios casados sem geometria na malha: {len(sem_malha)} {sorted(sem_malha)[:10]}")
    log(f"Feições da malha sem correspondência na base: {len(malha_sem_base)}")

    produtos = montar_produtos(base)
    clima = montar_clima_base(base, municipios)
    combos = (
        base[["cidade", "estado", "produto"]].drop_duplicates()
        .merge(municipios[["codigo_ibge", "cidade", "estado"]], on=["cidade", "estado"], how="inner")
    )[["codigo_ibge", "produto"]].sort_values(["codigo_ibge", "produto"]).reset_index(drop=True)
    stats, valores = montar_referencia(base)

    categorias = {c: sorted(base[c].dropna().unique().tolist()) for c in config.CATEGORICAS}
    faixas = {c: {"min": float(base[c].min()), "max": float(base[c].max())} for c in config.CLIMA}

    log("\n== Tabelas ==")
    log(f"municipios: {len(municipios)} | produtos: {len(produtos)} | clima_base: {len(clima)} | "
        f"combinacoes_validas: {len(combos)} | referencia_produto: {len(stats)}")
    comparar_categorias_modelo(categorias)

    municipios.to_csv(config.ARQ_MUNICIPIOS, index=False, encoding="utf-8")
    produtos.to_csv(config.ARQ_PRODUTOS, index=False, encoding="utf-8")
    clima.to_csv(config.ARQ_CLIMA_BASE, index=False, encoding="utf-8")
    combos.to_csv(config.ARQ_COMBINACOES, index=False, encoding="utf-8")
    stats.to_csv(config.ARQ_REF_PRODUTO, index=False, encoding="utf-8")
    valores.to_parquet(config.ARQ_REF_VALORES, index=False)
    config.ARQ_CATEGORIAS.write_text(json.dumps(categorias, ensure_ascii=False, indent=1), encoding="utf-8")
    config.ARQ_FAIXAS_CLIMA.write_text(json.dumps(faixas, ensure_ascii=False, indent=1), encoding="utf-8")
    config.ARQ_MALHA.write_text(json.dumps(malha, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    # Nomes de todas as feições da malha (para tooltip), priorizando a grafia da base.
    reg_uf = regiao_por_uf(municipios, ibge)
    nomes = pd.DataFrame({"codigo_ibge": sorted(cod_malha)}).merge(
        ibge[["codigo_ibge", "nome_ibge", "uf", "regiao_ibge"]], on="codigo_ibge", how="left"
    )
    nomes["nome_exibicao"] = nomes["nome_ibge"].fillna(nomes["codigo_ibge"]) + " - " + nomes["uf"].fillna("?")
    base_nomes = municipios.set_index("codigo_ibge")["nome_exibicao"]
    nomes.loc[nomes["codigo_ibge"].isin(base_nomes.index), "nome_exibicao"] = nomes["codigo_ibge"].map(base_nomes)
    nomes["na_base"] = nomes["codigo_ibge"].isin(base_nomes.index)
    # Região de cada feição (grafia da base, via UF), cobrindo também os municípios sem histórico.
    nomes["regiao"] = nomes["uf"].map(reg_uf).fillna(nomes["regiao_ibge"])
    nomes[["codigo_ibge", "nome_exibicao", "uf", "regiao", "regiao_ibge", "na_base"]].to_csv(
        config.ARQ_NOMES_MALHA, index=False, encoding="utf-8"
    )

    estados, regioes = montar_estados_regioes(malha, ibge, reg_uf)
    config.ARQ_MALHA_ESTADOS.write_text(json.dumps(estados, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    config.ARQ_MALHA_REGIOES.write_text(json.dumps(regioes, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    log(f"\nArquivos gravados em {config.DIR_REF}")
    Path(config.ARQ_RELATORIO).write_text("\n".join(LINHAS_RELATORIO), encoding="utf-8")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
