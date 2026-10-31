"""Carga do modelo, montagem das features, predição e verificação de ambiente.

Pontos únicos (importante para auditoria):
- ``montar_features`` é a ÚNICA função que monta o DataFrame enviado ao modelo
  e o ÚNICO lugar que define a coluna ``ano`` (sempre ``ANO_REFERENCIA``, exceto
  no teste de fumaça da inicialização, que usa o ano real de cada linha).
- ``prever_rendimento`` é o ÚNICO lugar onde ``np.expm1`` é aplicado.

Este módulo não depende do Streamlit; o app envolve as funções com cache.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xgboost as xgb

import config
from index_utils import ReferenciaProduto, carregar_referencia, indice_relativo


# ---------------------------------------------------------------------------
# Erros
# ---------------------------------------------------------------------------
class CategoriaDesconhecidaError(ValueError):
    """Valor categórico fora da lista do treino (``categorias.json``)."""


class CombinacaoInvalidaError(ValueError):
    """Par município × produto sem histórico (fora de ``combinacoes_validas``)."""


class EsquemaInvalidoError(ValueError):
    """DataFrame de features incompatível com o modelo."""


class MunicipioSemDadosError(ValueError):
    """Município sem cadastro ou sem clima base."""


MSG_BLOQUEIO = "Este produto não consta no histórico deste município. Escolha outro produto ou outro município."


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------
@dataclass
class Referencias:
    """Todas as tabelas auxiliares geradas pelo ``build_reference.py``."""

    municipios: pd.DataFrame  # index = codigo_ibge
    produtos: pd.DataFrame  # index = produto
    clima_base: pd.DataFrame  # index = codigo_ibge
    combinacoes: pd.DataFrame  # colunas codigo_ibge, produto
    referencia: ReferenciaProduto
    categorias: dict[str, list[str]]
    faixas_clima: dict[str, dict[str, float]]
    combos_set: set[tuple[str, str]] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.combos_set = set(zip(self.combinacoes["codigo_ibge"], self.combinacoes["produto"]))

    def produtos_validos(self, codigo_ibge: str) -> list[str]:
        """Produtos com histórico no município (ordenados)."""
        return sorted(self.combinacoes.loc[self.combinacoes["codigo_ibge"] == codigo_ibge, "produto"])

    def municipios_validos(self, produto: str) -> list[str]:
        """Municípios (``codigo_ibge``) com histórico do produto."""
        return self.combinacoes.loc[self.combinacoes["produto"] == produto, "codigo_ibge"].tolist()

    def combinacao_valida(self, codigo_ibge: str, produto: str) -> bool:
        """``True`` se o par município × produto existe na base."""
        return (codigo_ibge, produto) in self.combos_set


def carregar_modelo(caminho=config.ARQ_MODELO) -> xgb.XGBRegressor:
    """Carrega o XGBRegressor salvo em JSON."""
    modelo = xgb.XGBRegressor()
    modelo.load_model(str(caminho))
    return modelo


def carregar_referencias() -> Referencias:
    """Lê as tabelas auxiliares (``codigo_ibge`` sempre como texto de 7 dígitos)."""
    faltando = [
        p.name
        for p in (
            config.ARQ_MUNICIPIOS, config.ARQ_PRODUTOS, config.ARQ_CLIMA_BASE, config.ARQ_COMBINACOES,
            config.ARQ_REF_PRODUTO, config.ARQ_REF_VALORES, config.ARQ_CATEGORIAS, config.ARQ_FAIXAS_CLIMA,
            config.ARQ_MALHA, config.ARQ_NOMES_MALHA,
        )
        if not p.exists()
    ]
    if faltando:
        raise FileNotFoundError(
            "Arquivos auxiliares ausentes em data/ref/: " + ", ".join(faltando)
            + ". Rode antes: python build_reference.py"
        )
    dt = {"codigo_ibge": str}
    mun = pd.read_csv(config.ARQ_MUNICIPIOS, dtype=dt, encoding="utf-8").set_index("codigo_ibge", drop=False)
    prod = pd.read_csv(config.ARQ_PRODUTOS, encoding="utf-8").set_index("produto", drop=False)
    clima = pd.read_csv(config.ARQ_CLIMA_BASE, dtype=dt, encoding="utf-8").set_index("codigo_ibge", drop=False)
    combos = pd.read_csv(config.ARQ_COMBINACOES, dtype=dt, encoding="utf-8")
    categorias = json.loads(config.ARQ_CATEGORIAS.read_text(encoding="utf-8"))
    faixas = json.loads(config.ARQ_FAIXAS_CLIMA.read_text(encoding="utf-8"))
    return Referencias(mun, prod, clima, combos, carregar_referencia(), categorias, faixas)


# ---------------------------------------------------------------------------
# Montagem das features (ponto ÚNICO)
# ---------------------------------------------------------------------------
def montar_features(
    entrada: pd.DataFrame,
    refs: Referencias,
    usar_ano_da_entrada: bool = False,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Monta o DataFrame de features exatamente como o modelo espera.

    ``entrada`` precisa ter ``regiao``, ``estado``, ``cidade``, ``produto`` e as 6
    variáveis de clima. ``tipo_lavoura`` é SEMPRE derivado do produto (tabela
    ``produtos``); qualquer valor recebido é ignorado.

    ``ano``: preenchido com ``config.ANO_REFERENCIA``. Somente o teste de fumaça
    da inicialização passa ``usar_ano_da_entrada=True`` para usar o ano real.

    Raises:
        CategoriaDesconhecidaError: valor categórico fora de ``categorias.json``.
        EsquemaInvalidoError: colunas ausentes, valores não finitos ou ordem divergente.
    """
    features = features or config.FEATURES
    obrig = ["regiao", "estado", "cidade", "produto"] + config.CLIMA + (["ano"] if usar_ano_da_entrada else [])
    faltando = [c for c in obrig if c not in entrada.columns]
    if faltando:
        raise EsquemaInvalidoError(f"Colunas ausentes para montar as features: {faltando}")

    X = pd.DataFrame(index=entrada.index)
    if usar_ano_da_entrada:
        X["ano"] = entrada["ano"].astype("int64")
    else:
        X["ano"] = np.int64(config.ANO_REFERENCIA)
    for c in ("regiao", "estado", "cidade", "produto"):
        X[c] = entrada[c].astype(object)
    desconhecidos = sorted(set(X["produto"]) - set(refs.produtos.index))
    if desconhecidos:
        raise CategoriaDesconhecidaError(f"Produto desconhecido: {desconhecidos[:5]}")
    X["tipo_lavoura"] = X["produto"].map(refs.produtos["tipo_lavoura"])
    for c in config.CLIMA:
        X[c] = pd.to_numeric(entrada[c], errors="coerce").astype("float64")

    for c in config.CATEGORICAS:
        cats = refs.categorias[c]
        fora = sorted(set(X[c].dropna()) - set(cats)) + (["<nulo>"] if X[c].isna().any() else [])
        if fora:
            raise CategoriaDesconhecidaError(f"Categoria desconhecida em '{c}': {fora[:5]}")
        X[c] = X[c].astype(pd.CategoricalDtype(categories=cats))

    num = X[config.CLIMA].to_numpy()
    if not np.isfinite(num).all():
        raise EsquemaInvalidoError("Há valores de clima nulos ou não finitos.")

    X = X[features]
    validar_esquema(X, features)
    return X


def validar_esquema(X: pd.DataFrame, features: list[str] | None = None) -> None:
    """Confere ordem das colunas e dtypes antes do ``predict``."""
    features = features or config.FEATURES
    if list(X.columns) != list(features):
        raise EsquemaInvalidoError(f"Ordem/conjunto de features divergente: {list(X.columns)}")
    for c in features:
        if c in config.CATEGORICAS:
            if not isinstance(X[c].dtype, pd.CategoricalDtype):
                raise EsquemaInvalidoError(f"'{c}' deveria ser categórica")
        elif not pd.api.types.is_numeric_dtype(X[c]):
            raise EsquemaInvalidoError(f"'{c}' deveria ser numérica")


# ---------------------------------------------------------------------------
# Predição (ponto ÚNICO do expm1)
# ---------------------------------------------------------------------------
def prever_rendimento(modelo: xgb.XGBRegressor, X: pd.DataFrame) -> np.ndarray:
    """Prediz ``rendimento_ha`` em ESCALA ORIGINAL.

    O modelo foi treinado com ``log1p(rendimento_ha)``. Aqui — e somente aqui —
    a saída é revertida com ``np.expm1`` e limitada a >= 0.
    """
    validar_esquema(X)
    pred_log = modelo.predict(X)
    return np.clip(np.expm1(pred_log.astype("float64")), 0.0, None)  # <<< ÚNICO expm1 do app


# ---------------------------------------------------------------------------
# Cenários de clima
# ---------------------------------------------------------------------------
def aplicar_ajustes(clima: pd.DataFrame, ajustes: dict[str, float]) -> pd.DataFrame:
    """Aplica ajustes de cenário (percentuais ou absolutos, conforme ``config.AJUSTES_CLIMA``)."""
    out = clima.copy()
    for var, delta in ajustes.items():
        if not delta:
            continue
        tipo = config.AJUSTES_CLIMA[var][0]
        out[var] = out[var] * (1 + delta / 100.0) if tipo == "pct" else out[var] + delta
    out["chuva_total"] = out["chuva_total"].clip(lower=0)
    out["radiacao_solar_total"] = out["radiacao_solar_total"].clip(lower=0)
    out["umidade_ar_media"] = out["umidade_ar_media"].clip(0, 100)
    out["umidade_solo_media"] = out["umidade_solo_media"].clip(0, 1)
    return out


def fora_da_faixa(clima: pd.DataFrame, refs: Referencias) -> dict[str, int]:
    """Quantas linhas de cada variável ficam fora da faixa observada na base."""
    res = {}
    for var in config.CLIMA:
        f = refs.faixas_clima[var]
        n = int(((clima[var] < f["min"]) | (clima[var] > f["max"])).sum())
        if n:
            res[var] = n
    return res


def _linhas(refs: Referencias, pares: pd.DataFrame, clima: pd.DataFrame | None = None) -> pd.DataFrame:
    """Junta (codigo_ibge, produto) com cadastro do município e clima (base, se não informado)."""
    mun = refs.municipios[["regiao", "estado", "cidade", "nome_exibicao", "uf"]]
    clima = refs.clima_base[config.CLIMA] if clima is None else clima
    df = pares.join(mun, on="codigo_ibge", how="inner").join(clima, on="codigo_ibge", how="inner")
    return df


# ---------------------------------------------------------------------------
# Modos
# ---------------------------------------------------------------------------
def prever_mapa_produto(
    modelo: xgb.XGBRegressor,
    refs: Referencias,
    produto: str,
    ajustes: dict[str, float],
    valor_unico: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Modo 1: predição em lote (um único ``predict``) para todos os municípios com o produto."""
    cods = refs.municipios_validos(produto)
    pares = pd.DataFrame({"codigo_ibge": cods, "produto": produto})
    if valor_unico is not None:
        clima = pd.DataFrame([valor_unico] * len(cods), index=pd.Index(cods, name="codigo_ibge"))
    else:
        clima = aplicar_ajustes(refs.clima_base.loc[cods, config.CLIMA], ajustes)
    df = _linhas(refs, pares, clima)
    if df.empty:
        return df.assign(rendimento_previsto=[], indice=[])
    X = montar_features(df, refs)
    df["rendimento_previsto"] = prever_rendimento(modelo, X)
    df["indice"] = indice_relativo(df["produto"], df["rendimento_previsto"].to_numpy(), refs.referencia)
    return df.reset_index(drop=True)


def prever_produtos_municipios(
    modelo: xgb.XGBRegressor, refs: Referencias, codigos: list[str], ajustes: dict[str, float]
) -> pd.DataFrame:
    """Modo 2: todos os produtos válidos de um ou mais municípios, em um único ``predict``."""
    pares = refs.combinacoes[refs.combinacoes["codigo_ibge"].isin(codigos)].reset_index(drop=True)
    if pares.empty:
        return pd.DataFrame(columns=["codigo_ibge", "produto", "rendimento_previsto", "indice"])
    clima = aplicar_ajustes(refs.clima_base.loc[sorted(set(pares["codigo_ibge"])), config.CLIMA], ajustes)
    df = _linhas(refs, pares, clima)
    X = montar_features(df, refs)
    df["rendimento_previsto"] = prever_rendimento(modelo, X)
    df["indice"] = indice_relativo(df["produto"], df["rendimento_previsto"].to_numpy(), refs.referencia)
    df["tipo_lavoura"] = df["produto"].map(refs.produtos["tipo_lavoura"])
    return df.reset_index(drop=True)


def ranking_agregado(df: pd.DataFrame) -> pd.DataFrame:
    """Modo 2 (região/estado): mediana simples, por produto, entre os municípios da área."""
    if df.empty:
        return pd.DataFrame(columns=["produto", "tipo_lavoura", "indice", "rendimento_previsto", "n_municipios"])
    g = df.groupby(["produto", "tipo_lavoura"], as_index=False).agg(
        indice=("indice", "median"),
        rendimento_previsto=("rendimento_previsto", "median"),
        n_municipios=("codigo_ibge", "nunique"),
    )
    return g.sort_values("indice", ascending=False).reset_index(drop=True)


def prever_ponto(
    modelo: xgb.XGBRegressor, refs: Referencias, codigo_ibge: str, produto: str, clima: dict[str, float]
) -> dict:
    """Modo 3: rendimento previsto para um município, um produto e um clima informado.

    Raises:
        MunicipioSemDadosError: município fora da tabela ``municipios``.
        CombinacaoInvalidaError: par fora de ``combinacoes_validas`` (o modelo NÃO é chamado).
    """
    if codigo_ibge not in refs.municipios.index:
        raise MunicipioSemDadosError(f"Município {codigo_ibge} sem dados na base.")
    if not refs.combinacao_valida(codigo_ibge, produto):  # validação no servidor, antes do predict
        raise CombinacaoInvalidaError(MSG_BLOQUEIO)
    m = refs.municipios.loc[codigo_ibge]
    entrada = pd.DataFrame(
        [{"regiao": m["regiao"], "estado": m["estado"], "cidade": m["cidade"], "produto": produto,
          **{v: float(clima[v]) for v in config.CLIMA}}]
    )
    X = montar_features(entrada, refs)
    rend = float(prever_rendimento(modelo, X)[0])
    idx = float(indice_relativo([produto], np.array([rend]), refs.referencia)[0])
    return {
        "rendimento_previsto": rend,
        "indice": idx,
        "tipo_lavoura": str(X["tipo_lavoura"].iloc[0]),
        "ano": int(X["ano"].iloc[0]),
        "fora_faixa": fora_da_faixa(entrada, refs),
    }


# ---------------------------------------------------------------------------
# Verificação de ambiente
# ---------------------------------------------------------------------------
@dataclass
class ResultadoValidacao:
    """Resultado de ``validar_ambiente``: erros bloqueiam o app; avisos não."""

    erros: list[str] = field(default_factory=list)
    avisos: list[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.erros


def validar_ambiente(
    modelo: xgb.XGBRegressor,
    refs: Referencias,
    base: pd.DataFrame | None = None,
    features: list[str] | None = None,
    ano_referencia: int | None = None,
) -> ResultadoValidacao:
    """Checagens de inicialização (ver guia, seção "VERIFICAÇÃO NA INICIALIZAÇÃO").

    Os parâmetros ``features`` e ``ano_referencia`` existem para permitir testes
    de divergência; o app usa os valores de ``config``.
    """
    res = ResultadoValidacao()
    features = list(features or config.FEATURES)
    ano_referencia = config.ANO_REFERENCIA if ano_referencia is None else ano_referencia
    booster = modelo.get_booster()

    # 1. Nomes e ordem das features
    if list(booster.feature_names or []) != features:
        res.erros.append(
            f"As features do modelo {booster.feature_names} diferem (nome/ordem) de FEATURES em config.py {features}."
        )
    # 2. Tipos categóricos
    tipos = dict(zip(booster.feature_names or [], booster.feature_types or []))
    cat_modelo = sorted(f for f, t in tipos.items() if t == "c")
    if cat_modelo != sorted(config.CATEGORICAS):
        res.erros.append(f"Categóricas no modelo {cat_modelo} diferem das esperadas {sorted(config.CATEGORICAS)}.")

    if base is None:
        base = pd.read_csv(config.ARQ_BASE, encoding="utf-8")

    # 3. categorias.json
    for c in config.CATEGORICAS:
        lista = refs.categorias.get(c, [])
        if len(lista) != len(set(lista)):
            res.erros.append(f"categorias.json: '{c}' tem valores duplicados.")
        ausentes = set(base[c].dropna().unique()) - set(lista)
        if ausentes:
            res.erros.append(f"categorias.json: '{c}' não contém {sorted(ausentes)[:5]} (total {len(ausentes)}).")
    try:  # categorias gravadas no próprio modelo (xgboost >= 3.1)
        gravadas = dict(booster.get_categories(export_to_arrow=True).to_arrow())
        for c in config.CATEGORICAS:
            arr = gravadas.get(c)
            if arr is not None and arr.to_pylist() != refs.categorias.get(c):
                res.erros.append(f"As categorias de '{c}' gravadas no modelo diferem de categorias.json.")
        res.info["categorias_no_modelo"] = True
    except Exception:  # noqa: BLE001
        res.info["categorias_no_modelo"] = False

    # 4. Produto -> tipo único
    tipos_prod = base.groupby("produto")["tipo_lavoura"].nunique()
    ambiguos = tipos_prod[tipos_prod > 1].index.tolist()
    if ambiguos:
        res.erros.append(f"Produtos com mais de um tipo_lavoura na base: {ambiguos}")

    # 5. Ano de referência
    ano_max = int(base["ano"].max())
    res.info["ano_max_base"] = ano_max
    if ano_referencia != ano_max:
        res.erros.append(f"ANO_REFERENCIA ({ano_referencia}) difere do maior ano da base ({ano_max}).")

    if res.erros:  # não adianta seguir para o teste de fumaça
        return res

    # 6. Teste de fumaça (mesma função de montagem, com o ano real de cada linha)
    amostra = base.sample(n=min(config.N_AMOSTRA_FUMACA, len(base)), random_state=config.SEED)
    try:
        X = montar_features(amostra, refs, usar_ano_da_entrada=True, features=features)
        pred = prever_rendimento(modelo, X)
    except Exception as e:  # noqa: BLE001
        res.erros.append(f"Teste de fumaça falhou ao montar/predizer: {e}")
        return res
    if not np.isfinite(pred).all() or (pred < 0).any():
        res.erros.append("Teste de fumaça: há predições não finitas ou negativas.")
        return res
    real = amostra[config.ALVO].reset_index(drop=True)
    rho = float(real.rank().corr(pd.Series(pred).rank()))  # Spearman = Pearson dos postos
    res.info["spearman_fumaca"] = rho
    if not rho >= config.LIMIAR_SPEARMAN:
        res.avisos.append(
            f"Spearman entre real e previsto na amostra = {rho:.3f} (< {config.LIMIAR_SPEARMAN}). "
            "Possível divergência de categorias ou de ordem das features."
        )
    return res
