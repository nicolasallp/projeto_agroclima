"""Índice relativo (0–100) por produto.

O índice é o percentil do rendimento previsto (JÁ em escala original, após o
``expm1``) dentro da distribuição histórica nacional de ``rendimento_ha`` do
mesmo produto (``referencia_produto``). Escala log nunca entra aqui.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import config


@dataclass(frozen=True)
class ReferenciaProduto:
    """Distribuição histórica (escala original) de ``rendimento_ha`` por produto."""

    estatisticas: pd.DataFrame  # index = produto; colunas n, mediana, p10, p25, p75, p90, media
    valores_ordenados: dict[str, np.ndarray]  # produto -> array ordenado de rendimento_ha

    def stats(self, produto: str) -> pd.Series:
        """Estatísticas de um produto (``KeyError`` se o produto não existir)."""
        return self.estatisticas.loc[produto]


def carregar_referencia(
    arq_stats=config.ARQ_REF_PRODUTO, arq_valores=config.ARQ_REF_VALORES
) -> ReferenciaProduto:
    """Lê as estatísticas e os arrays ordenados gerados pelo ``build_reference.py``."""
    stats = pd.read_csv(arq_stats, encoding="utf-8").set_index("produto")
    vals = pd.read_parquet(arq_valores)
    ordenados = {
        str(p): np.sort(g[config.ALVO].to_numpy(dtype=float)) for p, g in vals.groupby("produto", sort=False)
    }
    return ReferenciaProduto(stats, ordenados)


def percentil_em(valores_ordenados: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Percentil (0–100) de cada ``x`` em um array ordenado, tratando empates pela média.

    Equivale a ``scipy.stats.percentileofscore(..., kind="mean")``: metade dos
    valores iguais conta como "abaixo".
    """
    n = len(valores_ordenados)
    if n == 0:
        return np.full(np.shape(x), np.nan)
    x = np.asarray(x, dtype=float)
    abaixo = np.searchsorted(valores_ordenados, x, side="left")
    ate = np.searchsorted(valores_ordenados, x, side="right")
    return (abaixo + ate) / 2.0 / n * 100.0


def indice_relativo(produtos: pd.Series | np.ndarray, rendimento: np.ndarray, ref: ReferenciaProduto) -> np.ndarray:
    """Calcula o índice relativo, vetorizado, para pares (produto, rendimento previsto).

    ``rendimento`` deve estar em ESCALA ORIGINAL (saída de ``model_utils.prever_rendimento``).
    """
    produtos = np.asarray(produtos, dtype=object)
    rendimento = np.asarray(rendimento, dtype=float)
    saida = np.full(len(rendimento), np.nan)
    for p in pd.unique(produtos):
        mask = produtos == p
        saida[mask] = percentil_em(ref.valores_ordenados.get(str(p), np.array([])), rendimento[mask])
    return saida


def texto_explicacao_indice() -> str:
    """Explicação curta, em linguagem simples, do índice relativo."""
    return (
        "**Índice relativo (0–100):** posição do rendimento previsto em relação a todo o "
        "histórico nacional do mesmo produto. Índice 80 significa que a previsão supera cerca "
        "de 80% dos rendimentos já registrados para esse produto no Brasil. Ele permite comparar "
        "produtos diferentes, cujos rendimentos têm escalas distintas. O modelo estima rendimento, "
        "não probabilidade: “mais propenso” quer dizer índice alto."
    )
