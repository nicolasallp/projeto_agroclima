"""Configurações centrais do AgroClima (caminhos, features e parâmetros fixos).

Tudo o que é "constante de negócio" fica aqui, para que app, utilitários e
``build_reference.py`` usem exatamente os mesmos valores.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
RAIZ: Path = Path(__file__).resolve().parent
ARQ_MODELO: Path = RAIZ / "modelo_anual.json"
ARQ_BASE: Path = RAIZ / "agroclima_anual.csv"
ARQ_DE_PARA: Path = RAIZ / "de_para_municipios.csv"

DIR_DATA: Path = RAIZ / "data"
DIR_IBGE: Path = DIR_DATA / "ibge"
DIR_REF: Path = DIR_DATA / "ref"

# Arquivos baixados do IBGE (reutilizados entre execuções)
ARQ_IBGE_MUNICIPIOS: Path = DIR_IBGE / "municipios_ibge.json"
ARQ_IBGE_MALHA: Path = DIR_IBGE / "malha_municipios_ibge.geojson"

# Tabelas auxiliares geradas por build_reference.py
ARQ_MODELO_UBJ: Path = DIR_REF / "modelo_anual.ubj.gz"
ARQ_MUNICIPIOS: Path = DIR_REF / "municipios.csv"
ARQ_PRODUTOS: Path = DIR_REF / "produtos.csv"
ARQ_CLIMA_BASE: Path = DIR_REF / "clima_base.csv"
ARQ_COMBINACOES: Path = DIR_REF / "combinacoes_validas.csv"
ARQ_REF_PRODUTO: Path = DIR_REF / "referencia_produto.csv"
ARQ_REF_VALORES: Path = DIR_REF / "referencia_valores.parquet"
ARQ_CATEGORIAS: Path = DIR_REF / "categorias.json"
ARQ_FAIXAS_CLIMA: Path = DIR_REF / "faixas_clima.json"
ARQ_MALHA: Path = DIR_REF / "malha_municipios.geojson"
ARQ_MALHA_ESTADOS: Path = DIR_REF / "malha_estados.geojson"  # 27 UFs (dissolvidas da malha municipal)
ARQ_MALHA_REGIOES: Path = DIR_REF / "malha_regioes.geojson"  # 5 regiões (dissolvidas dos estados)
ARQ_NOMES_MALHA: Path = DIR_REF / "nomes_malha.csv"  # nome/UF de todas as feições (inclusive sem histórico)
ARQ_RELATORIO: Path = DIR_REF / "relatorio_build.txt"

# ---------------------------------------------------------------------------
# URLs do IBGE (únicas fontes externas permitidas)
# ---------------------------------------------------------------------------
URL_IBGE_MUNICIPIOS: str = "https://servicodados.ibge.gov.br/api/v1/localidades/municipios"
URL_IBGE_MALHA_TEMPLATE: str = (
    "https://servicodados.ibge.gov.br/api/v3/malhas/paises/BR"
    "?formato=application/vnd.geo+json&qualidade={qualidade}&intrarregiao=municipio"
)
QUALIDADES_MALHA: tuple[str, ...] = ("minima", "1", "intermediaria")
MALHA_N_ESPERADO: int = 5570
MALHA_TOLERANCIA_SIMPLIFICACAO: float = 0.005  # graus (~500 m); só usada se a malha vier pesada
MALHA_TAMANHO_MAX_MB: float = 8.0
# Dissolução de estados/regiões (build_reference.py)
MALHA_BUFFER_FECHAMENTO: float = 0.002  # graus (~200 m): buffer(+eps).buffer(-eps) fecha frestas entre municípios
MALHA_AREA_MIN_FURO: float = 1e-3  # graus²: furos menores são frestas (o DF, furo real em GO, tem ~0,5)
MALHA_AGREGADA_MAX_MB: float = 1.5  # acima disso, estados/regiões são simplificados

# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
# ATENÇÃO: ordem EXATA gravada no modelo (booster.feature_names).
# O guia listava 'produto' antes de 'tipo_lavoura'; o modelo exige o contrário.
FEATURES: list[str] = [
    "ano",
    "regiao",
    "estado",
    "cidade",
    "tipo_lavoura",
    "produto",
    "chuva_total",
    "radiacao_solar_total",
    "umidade_ar_media",
    "umidade_solo_media",
    "temperatura_solo_media",
    "temperatura_media",
]
CATEGORICAS: list[str] = ["regiao", "estado", "cidade", "tipo_lavoura", "produto"]
CLIMA: list[str] = [
    "chuva_total",
    "radiacao_solar_total",
    "umidade_ar_media",
    "umidade_solo_media",
    "temperatura_solo_media",
    "temperatura_media",
]
ALVO: str = "rendimento_ha"

# Ano usado em TODAS as predições do app (maior ano da base; confirmado na inicialização).
ANO_REFERENCIA: int = 2024
# Janela (em anos) usada para o clima base de cada município.
ANOS_CLIMA_BASE: int = 10

# Verificação de inicialização
N_AMOSTRA_FUMACA: int = 2000
SEED: int = 42
LIMIAR_SPEARMAN: float = 0.8

# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
UNIDADE: str = "rendimento/ha"
TOP_N_PADRAO: int = 20
ESTILO_MAPA: str = "carto-positron"
CENTRO_BRASIL: dict[str, float] = {"lat": -14.5, "lon": -52.0}
ZOOM_BRASIL: float = 3.0

ROTULOS_CLIMA: dict[str, str] = {
    "chuva_total": "Chuva total (mm/ano)",
    "radiacao_solar_total": "Radiação solar total (MJ/m²)",
    "umidade_ar_media": "Umidade do ar média (%)",
    "umidade_solo_media": "Umidade do solo média (m³/m³)",
    "temperatura_solo_media": "Temperatura do solo média (°C)",
    "temperatura_media": "Temperatura média (°C)",
}

# Ajustes de cenário (Modos 1 e 2): (tipo, mínimo, máximo, passo, rótulo)
# tipo "pct" = variação percentual; "abs" = soma direta na unidade da variável.
AJUSTES_CLIMA: dict[str, tuple[str, float, float, float, str]] = {
    "chuva_total": ("pct", -50.0, 50.0, 5.0, "Chuva total (± %)"),
    "radiacao_solar_total": ("pct", -20.0, 20.0, 1.0, "Radiação solar total (± %)"),
    "umidade_ar_media": ("abs", -20.0, 20.0, 1.0, "Umidade do ar média (± pontos percentuais)"),
    "umidade_solo_media": ("pct", -50.0, 50.0, 5.0, "Umidade do solo média (± %)"),
    "temperatura_solo_media": ("abs", -5.0, 5.0, 0.5, "Temperatura do solo média (± °C)"),
    "temperatura_media": ("abs", -5.0, 5.0, 0.5, "Temperatura média (± °C)"),
}

# Campos numéricos do Modo 3: (passo, formato)
FORMATO_CLIMA: dict[str, tuple[float, str]] = {
    "chuva_total": (10.0, "%.1f"),
    "radiacao_solar_total": (10.0, "%.2f"),
    "umidade_ar_media": (0.5, "%.2f"),
    "umidade_solo_media": (0.01, "%.3f"),
    "temperatura_solo_media": (0.1, "%.2f"),
    "temperatura_media": (0.1, "%.2f"),
}

MODOS: list[str] = ["Onde produzir?", "O que produzir?", "Qual será o rendimento?"]

# Modo 2: nível do mapa e cores fixas por região (paleta Okabe-Ito, segura para daltônicos).
# As chaves são o nome NORMALIZADO (geo_utils.normalizar_nome), pois a grafia da base pode diferir da do IBGE.
NIVEIS_M2: list[str] = ["Cidades", "Estados", "Regiões"]
CORES_REGIOES: dict[str, str] = {
    "norte": "#009E73",
    "nordeste": "#E69F00",
    "centro oeste": "#F0E442",
    "sudeste": "#0072B2",
    "sul": "#CC79A7",
}
COR_SEM_REGIAO: str = "#bdbdbd"
NOTA_ANO: str = f"Estimativa nas condições do período mais recente da base ({ANO_REFERENCIA})."
NOTA_ESTIMATIVA: str = (
    "O valor exibido é uma estimativa típica (próxima da mediana) e tende a ficar "
    "um pouco abaixo da média real."
)
