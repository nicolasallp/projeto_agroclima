"""Testes de model_utils, index_utils e geo_utils.

Rodar:  python -m pytest -q
(Requer ter executado ``python build_reference.py`` antes.)
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import config
import geo_utils as geo
import model_utils as mu
from index_utils import percentil_em

RAIZ = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def modelo():
    return mu.carregar_modelo()


@pytest.fixture(scope="session")
def refs():
    return mu.carregar_referencias()


@pytest.fixture(scope="session")
def base():
    return pd.read_csv(config.ARQ_BASE, encoding="utf-8")


class Espiao:
    """Envolve ``modelo.predict`` e guarda cada DataFrame recebido."""

    def __init__(self, modelo):
        self.modelo = modelo
        self.original = modelo.predict
        self.chamadas: list[pd.DataFrame] = []

    def __enter__(self):
        def predict(X, *a, **k):
            self.chamadas.append(X.copy())
            return self.original(X, *a, **k)

        self.modelo.predict = predict
        return self

    def __exit__(self, *exc):
        self.modelo.predict = self.original


# ---------------------------------------------------------------------------
# Verificação de inicialização
# ---------------------------------------------------------------------------
def test_validacao_passa(modelo, refs, base):
    res = mu.validar_ambiente(modelo, refs, base=base)
    assert res.ok, res.erros
    assert res.info["ano_max_base"] == config.ANO_REFERENCIA
    assert res.info["spearman_fumaca"] >= config.LIMIAR_SPEARMAN


def test_validacao_aborta_com_ordem_trocada(modelo, refs, base):
    trocada = list(config.FEATURES)
    i, j = trocada.index("produto"), trocada.index("tipo_lavoura")
    trocada[i], trocada[j] = trocada[j], trocada[i]
    res = mu.validar_ambiente(modelo, refs, base=base, features=trocada)
    assert not res.ok
    assert any("diferem (nome/ordem)" in e for e in res.erros)


def test_validacao_aborta_com_ano_referencia_errado(modelo, refs, base):
    res = mu.validar_ambiente(modelo, refs, base=base, ano_referencia=2030)
    assert not res.ok and any("ANO_REFERENCIA" in e for e in res.erros)


# ---------------------------------------------------------------------------
# Ano de referência e expm1
# ---------------------------------------------------------------------------
def test_todas_as_predicoes_usam_ano_referencia(modelo, refs):
    cod = refs.municipios.index[0]
    prod = refs.produtos_validos(cod)[0]
    with Espiao(modelo) as esp:
        mu.prever_mapa_produto(modelo, refs, "Soja (em grão)", {"chuva_total": 10.0})
        mu.prever_mapa_produto(modelo, refs, "Milho (em grão)", {}, valor_unico=refs.clima_base[config.CLIMA].median().to_dict())
        mu.prever_produtos_municipios(modelo, refs, [cod], {})
        mu.prever_produtos_municipios(modelo, refs, refs.municipios.index[refs.municipios["uf"] == "SP"].tolist(), {})
        mu.prever_ponto(modelo, refs, cod, prod, refs.clima_base.loc[cod, config.CLIMA].to_dict())
    assert len(esp.chamadas) == 5
    for X in esp.chamadas:
        assert list(X.columns) == config.FEATURES
        assert (X["ano"] == config.ANO_REFERENCIA).all()


def test_expm1_em_um_unico_ponto():
    ocorrencias = []
    for arq in RAIZ.glob("*.py"):
        for n, linha in enumerate(arq.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"np\.expm1\(", linha):
                ocorrencias.append((arq.name, n))
    assert len(ocorrencias) == 1 and ocorrencias[0][0] == "model_utils.py", ocorrencias


def test_ano_definido_apenas_em_montar_features():
    """Nenhum outro arquivo do app atribui a coluna 'ano'."""
    for nome in ("app.py", "geo_utils.py", "index_utils.py"):
        texto = (RAIZ / nome).read_text(encoding="utf-8")
        assert not re.search(r"\[[\"']ano[\"']\]\s*=", texto), nome


def test_predicoes_nao_negativas_e_finitas(modelo, refs):
    df = mu.prever_mapa_produto(modelo, refs, "Café (em grão) Total", {})
    assert np.isfinite(df["rendimento_previsto"]).all() and (df["rendimento_previsto"] >= 0).all()
    assert df["indice"].between(0, 100).all()


# ---------------------------------------------------------------------------
# Categóricas e combinações
# ---------------------------------------------------------------------------
def test_combinacao_invalida_bloqueia_sem_chamar_predict(modelo, refs):
    cod = refs.municipios.index[0]
    invalido = next(p for p in refs.produtos.index if not refs.combinacao_valida(cod, p))
    with Espiao(modelo) as esp:
        with pytest.raises(mu.CombinacaoInvalidaError) as e:
            mu.prever_ponto(modelo, refs, cod, invalido, refs.clima_base.loc[cod, config.CLIMA].to_dict())
    assert esp.chamadas == []
    assert str(e.value) == mu.MSG_BLOQUEIO


def test_categoria_desconhecida(refs):
    entrada = pd.DataFrame([{"regiao": "Sul", "estado": "PR", "cidade": "Atlântida Perdida",
                             "produto": "Soja (em grão)", **{v: 1.0 for v in config.CLIMA}}])
    with pytest.raises(mu.CategoriaDesconhecidaError):
        mu.montar_features(entrada, refs)


def test_tipo_lavoura_sempre_derivado_do_produto(refs):
    entrada = pd.DataFrame([{"regiao": "Sul", "estado": "PR", "cidade": "Cascavel", "produto": "Soja (em grão)",
                             "tipo_lavoura": "Permanente", **{v: 1.0 for v in config.CLIMA}}])
    X = mu.montar_features(entrada, refs)
    assert X["tipo_lavoura"].iloc[0] == refs.produtos.loc["Soja (em grão)", "tipo_lavoura"]
    for c in config.CATEGORICAS:
        assert list(X[c].cat.categories) == refs.categorias[c]


def test_ranking_municipio_so_produtos_validos(modelo, refs):
    cod = refs.municipios.index[100]
    df = mu.prever_produtos_municipios(modelo, refs, [cod], {})
    assert sorted(df["produto"]) == refs.produtos_validos(cod)


# ---------------------------------------------------------------------------
# Município clicado = codigo_ibge correto (nomes repetidos)
# ---------------------------------------------------------------------------
def test_clique_em_cidade_repetida_usa_codigo(refs):
    mun = refs.municipios
    repetidos = mun[mun["cidade"] == "Alto Alegre"]
    assert len(repetidos) >= 2
    malha = geo.carregar_malha()
    nomes = pd.read_csv(config.ARQ_NOMES_MALHA, dtype={"codigo_ibge": str})
    alvo = repetidos.index[1]
    fig, curvas = geo.figura_mapa_selecao(malha, nomes, None, config.CENTRO_BRASIL, 3, "t")
    idx = curvas[0].index(alvo)
    customdata = fig.data[0].customdata[idx]
    # Evento com customdata
    assert geo.codigo_do_ponto({"curve_number": 0, "point_index": idx, "customdata": list(customdata)}) == alvo
    # Evento sem customdata: usa point_index na lista de códigos da curva
    assert geo.codigo_do_ponto({"curve_number": 0, "point_index": idx}, curvas) == alvo
    # Os dois "Alto Alegre" têm códigos distintos
    assert len(set(repetidos.index)) == len(repetidos)


def test_normalizacao():
    assert geo.normalizar_nome("  Alta Floresta D'Oeste ") == "alta floresta doeste"
    assert geo.normalizar_compacto("Embu-Guaçu") == geo.normalizar_compacto("Embu Guaçu")


# ---------------------------------------------------------------------------
# Índice relativo
# ---------------------------------------------------------------------------
def test_percentil_com_empates():
    v = np.array([1.0, 2.0, 2.0, 3.0])
    assert np.allclose(percentil_em(v, np.array([0.5, 2.0, 10.0])), [0.0, 50.0, 100.0])
