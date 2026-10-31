# AgroClima — Estimativa de Rendimento Agrícola por Hectare

Aplicação web em **Streamlit** que utiliza o modelo treinado (`modelo_anual.json`, `XGBRegressor` em `xgboost==3.4.1`) e a base histórica (`agroclima_anual.csv`, 2000–2024) para estimar o rendimento agrícola por hectare nos municípios brasileiros em três modos de uso:

1. **Onde produzir?** — variáveis de clima e produto → mapa coroplético e ranking Top N de municípios.
2. **O que produzir?** — clique em município no mapa (ou seleção de região/estado) → ranking de produtos ordenado pelo índice relativo (0–100).
3. **Qual será o rendimento?** — local, produto e variáveis climáticas → estimativa pontual de rendimento/ha, índice relativo e posição na distribuição histórica nacional.

---

## Estrutura de Pastas

```text
projeto_agroclima/
├── agroclima_anual.csv          # Base histórica (2000–2024, 1.436.395 linhas)
├── modelo_anual.json            # Modelo XGBRegressor treinado (xgboost 3.4.1)
├── de_para_municipios.csv       # De-para manual opcional para municípios (cidade, estado -> codigo_ibge)
├── config.py                    # Caminhos, constantes, features e parâmetros fixos
├── geo_utils.py                 # Normalização, manipulação da malha e mapas Choroplethmap (MapLibre)
├── index_utils.py               # Cálculo vetorizado do índice relativo (percentil 0–100 na escala original)
├── model_utils.py               # Carga, montagem única de features, predição (expm1 único) e validação
├── build_reference.py           # Script de geração das tabelas auxiliares e download da malha/IBGE
├── app.py                       # Interface Streamlit com os 3 modos e persistência em session_state
├── requirements.txt             # Dependências com versões fixadas
├── data/
│   ├── ibge/                    # Cache local dos downloads da API do IBGE
│   │   ├── municipios_ibge.json
│   │   └── malha_municipios_ibge.geojson
│   └── ref/                     # Tabelas de referência geradas por build_reference.py
│       ├── municipios.csv
│       ├── produtos.csv
│       ├── clima_base.csv
│       ├── combinacoes_validas.csv
│       ├── referencia_produto.csv
│       ├── referencia_valores.parquet
│       ├── categorias.json
│       ├── faixas_clima.json
│       ├── malha_municipios.geojson
│       ├── nomes_malha.csv
│       └── relatorio_build.txt
└── tests/                       # Testes automatizados (pytest + Streamlit AppTest)
    ├── conftest.py
    ├── test_model_utils.py
    └── test_app.py
```

---

## Como Executar

### 1. Instalar dependências
Recomenda-se utilizar um ambiente virtual Python:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Gerar as tabelas auxiliares e baixar a malha do IBGE
Execute uma vez antes de iniciar o app (os arquivos do IBGE ficam cacheados em `data/ibge/`):

```powershell
python build_reference.py
```

### 3. Iniciar a aplicação Streamlit

```powershell
streamlit run app.py
```

### 4. Executar a suíte de testes (opcional)

```powershell
pytest -q
```
