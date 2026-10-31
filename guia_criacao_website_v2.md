# PAPEL
Você é um engenheiro de software sênior e cientista de dados, especialista em Streamlit, visualização geoespacial e deploy de modelos de Machine Learning (XGBoost).

# OBJETIVO
Construir um website em Streamlit com três modos de uso, que usa o modelo treinado (arquivo em anexo) para estimar o rendimento agrícola por hectare nos municípios brasileiros: (1) mapa de onde produzir, (2) ranking de produtos por município clicado no mapa e (3) estimativa pontual de rendimento a partir de variáveis informadas pelo usuário. O app faz apenas inferência.

# ARQUIVOS ANEXADOS
- `modelo_anual.json` (modelo XGBoost)
- `agroclima_anual.csv` (base única, a mesma do treino)
Os dados do IBGE (municípios e malha) NÃO são anexados: o `build_reference.py` deve baixá-los (ver "DADOS AUXILIARES").

# CONTEXTO DO MODELO (não assuma nada além do que está aqui)
- Arquivo do modelo: modelo_anual.json
- Tipo: XGBRegressor (regressão). Versão do xgboost no treino: 3.4.1
- Alvo: `rendimento_ha`, já calculado na base. Sem scaler e sem normalização por produto.
- Transformação do alvo no treino: log1p(rendimento_ha). O modelo prevê em escala log. No app:
  1. Aplique `np.expm1` na saída de `predict` e limite o resultado a >= 0 antes de exibir qualquer valor.
  2. O valor exibido é uma estimativa típica (próxima da mediana condicional) e tende a ficar um pouco abaixo da média real. Informe isso em uma nota curta na interface.
  3. O índice relativo (percentil) deve ser calculado sobre `rendimento_ha` na escala ORIGINAL, comparando a predição já revertida com a distribuição histórica de `referencia_produto`. Nunca misture escala log com escala original.
- Features (12), na ordem exata do treino:
  ['ano', 'regiao', 'estado', 'cidade', 'produto', 'tipo_lavoura', 'chuva_total', 'radiacao_solar_total', 'umidade_ar_media', 'umidade_solo_media', 'temperatura_solo_media', 'temperatura_media']
- Categóricas (`regiao`, `estado`, `cidade`, `tipo_lavoura`, `produto`): NÃO são codificadas numericamente. `estado` é o nome completo (ex.: "São Paulo"), não a sigla. `tipo_lavoura` é função do produto: cada produto tem um único tipo, e esse valor vem da tabela `produtos`.
- Unidades do clima (granularidade anual): chuva_total em mm/ano; radiacao_solar_total em MJ/m²; umidade_ar_media em %; umidade_solo_media em m³/m³; temperatura_solo_media em °C; temperatura_media em °C.
- Faixa de anos do treino: 2000-2024.
- Tratamento do `ano` no app: o usuário NUNCA escolhe o ano, em nenhum modo. O `ano` é uma feature do modelo, mas o app é de previsão e o XGBoost não extrapola (anos acima de 2024 cairiam nas mesmas folhas de 2024). Por isso, toda predição usa `ANO_REFERENCIA = 2024` (o maior ano da base, definido em `config.py` e confirmado na inicialização). A coluna `ano` é preenchida por esse valor na ÚNICA função de montagem das features. O app não projeta crescimento tecnológico futuro: assume o nível do período mais recente da base. Não crie nenhum seletor, filtro ou campo de ano na interface.
- Base de dados única: agroclima_anual.csv, a mesma usada no treino. Nenhum filtro ou tratamento de outliers foi aplicado no treino; todos os filtros já estão no arquivo. O app e o `build_reference.py` NÃO devem aplicar filtro, remoção de linhas ou tratamento de outliers.
- Unidade do rendimento por produto: rótulo genérico "rendimento/ha". Os valores NÃO são comparáveis entre produtos.
- Pré-processamento das features numéricas: nenhum (não aplicar scaler).

Antes de escrever o app, inspecione o modelo (`get_booster().feature_names`, `feature_types`, `get_params()`) e confirme as features e quais são categóricas. Se algo divergir do descrito acima, sinalize e siga o que o modelo exige.

# REGRA CRÍTICA: CATEGÓRICAS
- Ao predizer, converta cada coluna categórica para `pd.CategoricalDtype(categories=<lista salva do treino>)`, com as mesmas categorias e a mesma ordem do treino. Salve essas listas em `categorias.json`, gerado a partir da base de treino.
- Os valores (grafia, acentos, caixa) devem ser idênticos aos do treino. Nunca envie categoria desconhecida ao modelo: restrinja o app às categorias conhecidas.
- `tipo_lavoura` nunca é escolhido livremente nem combinado com um produto de outro tipo: em todos os modos, é derivado de `produtos` a partir do produto. Nos Modos 1 e 3, o filtro "tipo de lavoura" só restringe a lista de produtos exibida.
- A grafia dos valores de `tipo_lavoura` é a da base, sem normalização.
- Nomes de cidade se repetem entre estados: nunca use o nome sozinho como chave.
- A chave interna dos municípios é `codigo_ibge` (7 dígitos, texto). Ele NÃO existe na base: vem do IBGE e é associado à base por cidade + estado normalizados (minúsculas, sem acentos, sem pontuação, espaços aparados). O `codigo_ibge` serve apenas para junção e mapa e nunca é enviado ao modelo.
- Reporte quantos municípios da base não casaram com o IBGE e quantos da malha ficaram sem correspondência.

# DADOS AUXILIARES
Gerar com `build_reference.py` a partir da base única, exceto `codigo_ibge`, `uf` e geometria, que vêm do IBGE e são BAIXADOS pelo script (sem arquivos anexados).

Fontes do IBGE (use estas URLs; não invente outras):
- Municípios (código, nome, UF, sigla, região): https://servicodados.ibge.gov.br/api/v1/localidades/municipios
- Malha municipal, GeoJSON: https://servicodados.ibge.gov.br/api/v3/malhas/paises/BR?formato=application/vnd.geo+json&qualidade=minima&intrarregiao=municipio
  (se `qualidade=minima` for rejeitada, teste `qualidade=1` ou `intermediaria`; se a API v3 falhar, consulte a documentação v4 em https://servicodados.ibge.gov.br/api/docs/malhas?versao=4).

Regras de download:
- Valide as respostas antes de usá-las: inspecione a estrutura real do JSON/GeoJSON e identifique em que campo está o código do município (na malha, provavelmente `properties.codarea`; renomeie para `codigo_ibge` como texto de 7 dígitos). Confirme que a malha tem ~5.570 feições.
- Na tabela de municípios, trate a hierarquia de UF e região pela estrutura real da resposta (alguns municípios podem não ter microrregião; use a região imediata como alternativa, se necessário).
- Salve os arquivos baixados em `data/ibge/` e reutilize-os nas execuções seguintes (baixe só se não existirem).
- Se o download falhar (sem internet ou API indisponível), o script deve parar com mensagem clara dizendo quais arquivos colocar manualmente em `data/ibge/` e com quais nomes. Não prossiga com dados parciais.
- Se a malha vier pesada, simplifique a geometria (ex.: geopandas/shapely `simplify` com tolerância pequena, preservando a topologia) para o mapa renderizar rápido.

Tabelas:
1. `municipios`: codigo_ibge (7 dígitos), regiao, estado (nome completo, grafia da base), cidade (grafia da base), uf (sigla, vinda do IBGE), nome_exibicao.
   - Junção base × IBGE por (cidade normalizada, estado normalizado). O IBGE fornece nome do estado e sigla, então a base não precisa de mapeamento nome → UF.
   - Para os não casados (grafias diferentes, municípios renomeados), aceite um `de_para_municipios.csv` manual e reporte os não casados nos dois sentidos.
   - Valide que cada par (cidade, estado) tem uma única `regiao`.
2. `produtos`: produto (grafia da base), tipo_lavoura (da coluna da base), unidade ("rendimento/ha").
3. `clima_base`: média das 6 variáveis de clima por município, calculada APENAS sobre os últimos 10 anos da base (2015-2024). Defina `ANOS_CLIMA_BASE = 10` em `config.py` e derive a janela do maior ano da base. Se um município não tiver nenhuma linha nessa janela, use a média de todos os anos disponíveis dele, marque-o com `clima_base_fallback = True` e reporte quantos municípios caíram nesse caso.
4. `combinacoes_validas`: pares município × produto presentes na base (todos os anos).
5. `referencia_produto`: por produto, a distribuição de `rendimento_ha` em ESCALA ORIGINAL, sem log (mediana, P10, P25, P75, P90 e array para percentil), calculada diretamente sobre a base fornecida (todos os anos), sem filtros adicionais (ignore apenas nulos no cálculo dos percentis).
6. `categorias.json`: listas de categorias do treino, para as 5 colunas categóricas.
7. Malha municipal (GeoJSON simplificado, ~5.570 municípios, chave `codigo_ibge` como texto de 7 dígitos).

Validação da base (sem alterá-la): reporte a contagem de nulos, infinitos e valores <= 0 em `rendimento_ha`, e de nulos nas features. NÃO remova nem corrija linhas: o modelo foi treinado com a base exatamente como está.

# COMPARABILIDADE ENTRE PRODUTOS
- O rendimento previsto só é comparável DENTRO de um mesmo produto.
- Rankings entre produtos (Modo 2) usam SEMPRE o ÍNDICE RELATIVO (0–100): percentil do rendimento previsto (já em escala original) na distribuição histórica nacional do mesmo produto (`referencia_produto`).
- Mapa e Top N do Modo 1 (um produto por vez) permitem alternar entre "Índice relativo (padrão)" e "Rendimento previsto (unidade)".
- No Modo 2, a tabela mostra o rendimento bruto com a unidade, mas a ordenação e o gráfico usam o índice. Nunca coloque valores brutos de produtos diferentes no mesmo eixo.
- Explique na interface, em linguagem simples, o que o índice significa. O modelo estima rendimento, não probabilidade: "mais propenso" = índice alto.

# FUNCIONALIDADES

- Navegação entre os modos por `st.segmented_control` (padrão "Onde produzir?"; se retornar None, volte ao último modo ativo guardado em `session_state`): "Onde produzir?", "O que produzir?" e "Qual será o rendimento?". Renderize APENAS o modo ativo. NÃO use `st.tabs`, que executa o código de todas as abas a cada interação e rodaria a predição do Modo 1 (~5.570 municípios) mesmo com o usuário em outro modo.
- Persistência: o Streamlit descarta o estado dos widgets que não são renderizados. Por isso, guarde os valores de cada modo (ajustes de clima, produto, município clicado, filtros) em chaves próprias de `session_state` com prefixo por modo (ex.: `m1_produto`, `m2_codigo_ibge`), copiadas por `on_change`, e use-as como padrão ao recriar os widgets. O estado de um modo não altera o dos outros.
- Não existe seletor de ano em nenhum modo (ver "Tratamento do `ano` no app"). Em cada modo, exiba uma nota fixa e curta: "Estimativa nas condições do período mais recente da base (2024)."
- A predição do Modo 1 deve ser cacheada (`st.cache_data`) por (produto, ajustes de clima), para que trocar de modo e voltar não a recalcule.

## Modo 1: "Onde produzir?" (variáveis → mapa)
Painel de controles no topo da aba (expander ou colunas):
- Produto (selectbox, com filtro prévio por tipo de lavoura: temporária/permanente). Mostre a unidade.
- Variáveis de clima: por padrão, cada município usa seu clima base dos últimos 10 anos (`clima_base`). O usuário aplica ajustes de cenário em cada uma das 6 variáveis (ex.: chuva ±%, temperatura ± °C). Opção avançada: valor único para todos os municípios.
Ação: predição em lote para todos os municípios com combinação válida para o produto escolhido, com `ano = ANO_REFERENCIA`.
Saída:
- Mapa coroplético com alternância entre "Índice relativo (padrão)" e "Rendimento previsto (unidade)".
- Municípios sem combinação válida em cinza, com legenda "sem histórico de produção".
- Tabela Top N (padrão 20): município, UF, rendimento previsto, unidade, índice; botão de download em CSV.

## Modo 2: "O que produzir?" (clique no mapa → produtos)
Layout em duas colunas (`st.columns([3, 2])`): mapa à ESQUERDA e painel de resultados à DIREITA. Em telas estreitas, as colunas empilham (mapa em cima, resultados abaixo).

Controles complementares (painel de controles no topo da aba, expander ou colunas, acima do mapa):
- Ajustes de clima, como no Modo 1.
- Seletores opcionais de região e estado, usados para (a) aproximar o mapa da área escolhida e (b) obter o ranking agregado da região/estado (mediana simples dos municípios, com o método explicado na interface). Quando uma área for escolhida por esse seletor, o painel direito mostra o ranking agregado; quando um município for clicado, mostra o do município. Deixe sempre claro, no cabeçalho do painel, qual dos dois está sendo exibido.

Mapa (esquerda):
- Mapa interativo do Brasil com todos os municípios (malha IBGE), com zoom e pan.
- O usuário seleciona um município CLICANDO nele. Use o mecanismo nativo de seleção do Streamlit (`st.plotly_chart(..., on_select="rerun")` ou equivalente) e identifique o município clicado pelo `codigo_ibge`, nunca pelo nome.
- Cor de base neutra (não representa predição). O município selecionado fica destacado (cor e contorno distintos).
- Preserve o zoom e a posição do mapa após o clique (ex.: `uirevision` no Plotly), para o mapa não "pular" ao atualizar.
- Guarde a seleção em `st.session_state` (ela persiste enquanto o usuário altera os ajustes de clima).
- Tooltip ao passar o mouse: município e UF.

Painel de resultados (direita), exibido após o clique:
- Cabeçalho: município, UF e região.
- Ranking dos produtos (gráfico de barras horizontais + tabela), conforme a regra de comparabilidade (índice relativo para ordenar e plotar; rendimento bruto com unidade apenas na tabela). Predição com `ano = ANO_REFERENCIA`.
- Considere apenas produtos com combinação válida no município.
- Antes do primeiro clique, mostre uma mensagem de orientação ("Clique em um município no mapa para ver os produtos mais propensos").
- Município sem nenhum produto válido: mensagem clara, sem erro.

## Modo 3: "Qual será o rendimento?" (variáveis → rendimento/ha)
O usuário informa as variáveis de entrada do modelo e recebe o rendimento por hectare previsto.

Seleção de local e produto (FORA do `st.form`, para que as listas dependentes atualizem a cada escolha):
- Filtros em cascata região → estado → município, com valores da tabela `municipios` (grafia do treino).
- Produto: o selectbox lista APENAS os produtos com combinação válida para o município escolhido (`combinacoes_validas`), com filtro opcional por tipo de lavoura. Mostre a unidade do produto.
- Só são aceitos valores categóricos presentes na base. Não permita digitação livre de município ou produto.

Restaurar clima: o botão "Restaurar clima base" NÃO fica dentro do `st.form` (o formulário só aceita `form_submit_button`; um `st.button` ali gera erro). Coloque-o logo acima do formulário, com `on_click=restaurar_clima`. A função grava os valores de `clima_base` do município nas chaves dos campos numéricos (que incluem o `codigo_ibge`). Faça isso no callback, pois o Streamlit não permite alterar o estado de um widget já criado na mesma execução.

Formulário (`st.form` com botão "Calcular", para não rodar o modelo a cada alteração):
- As 6 variáveis de clima, em campos numéricos com a unidade no rótulo: "Chuva total (mm/ano)", "Radiação solar total (MJ/m²)", "Umidade do ar média (%)", "Umidade do solo média (m³/m³)", "Temperatura do solo média (°C)", "Temperatura média (°C)". Use `step` e `format` adequados à unidade; em `umidade_solo_media` (varia entre 0 e 1), use `step=0.01` e `format="%.3f"`.
- Pré-preencha com o clima base dos últimos 10 anos do município (`clima_base`) e use uma chave de widget que inclua o `codigo_ibge`, para os valores reiniciarem ao trocar de município. O usuário pode editar livremente.
- Validação: cada campo tem limites mínimo e máximo derivados da faixa observada na base. Valores fora da faixa são aceitos, mas geram alerta de extrapolação (o modelo não extrapola bem).
- Use `form_submit_button(disabled=...)` para o bloqueio de combinação sem histórico.

Bloqueio de combinação sem histórico:
- Se a combinação município × produto não existir em `combinacoes_validas` (ou se o município não tiver nenhum produto válido), NÃO execute o modelo. Desabilite o botão "Calcular" e mostre a mensagem: "Este produto não consta no histórico deste município. Escolha outro produto ou outro município."
- Valide também no servidor, imediatamente antes do `predict`, e não confie apenas no estado da interface.
- Nunca envie ao modelo uma categoria fora de `categorias.json`.

Saída (após clicar em "Calcular"):
- Rendimento previsto por hectare, em destaque (`st.metric`), com a unidade do produto e após o `expm1`. Predição com `ano = ANO_REFERENCIA`.
- Índice relativo (0–100), com explicação curta.
- Contexto histórico do produto: faixa P10–P90 e mediana de `referencia_produto`, com a posição da previsão (gráfico de faixa horizontal de um único produto). Rotule como "distribuição histórica nacional deste produto". Não é intervalo de confiança da previsão.
- Alerta, quando aplicável: clima fora da faixa observada na base.
- Resumo dos valores de entrada usados, para conferência, incluindo o `tipo_lavoura` derivado do produto e o ano de referência (2024), exibido como parâmetro fixo e não editável.

Importante: a saída é rendimento por hectare, não volume total. A interface não deve prometer "quantidade total produzida" em nenhum rótulo ou texto.

# REQUISITOS TÉCNICOS
- Mapa: use `plotly.graph_objects.Choroplethmap` / `px.choropleth_map` (MapLibre, sem token; Plotly >= 5.24) com estilo "carto-positron". NÃO use `choropleth_mapbox`, que é legado. Use `featureidkey="properties.codigo_ibge"` com o código como texto de 7 dígitos em ambos os lados. Alternativa, se houver problema de fluidez ou de clique: pydeck (`GeoJsonLayer`) ou folium via `streamlit-folium`, com justificativa. Tooltip (município, UF, rendimento, unidade, índice), zoom e fluidez com ~5.570 municípios. Fixe plotly e streamlit (>= 1.40) no `requirements.txt`.
- Desempenho: `st.cache_resource` para o modelo; `st.cache_data` para tabelas e geometrias. Predição vetorizada em lote (um único `predict` por ação), nunca em loop por município.
- Montagem das features: DataFrame com as 12 colunas na ordem do modelo, dtypes corretos e validação de esquema antes de predizer. Use uma ÚNICA função de montagem em `model_utils.py` para todos os modos. É ela que preenche `ano` com `ANO_REFERENCIA`; nenhum outro ponto do código define o ano.
- Alerta na interface quando o cenário de clima sair da faixa observada na base.
- Erros: mensagens claras para município sem dados, produto sem histórico, categoria desconhecida e incompatibilidade de features.
- Interface em português do Brasil.
- Seleção por clique (Modo 2): exige Streamlit com suporte a seleção em gráficos. Teste que o clique funciona na camada de mapa escolhida com ~5.570 polígonos. Se a camada escolhida não retornar o município clicado de forma confiável, use `streamlit-folium` (`st_folium`, campo `last_active_drawing` / `last_object_clicked`) e justifique a troca.
- Ao clicar, a predição é feita apenas para o município selecionado (um único `predict` em lote com uma linha por produto válido), reaproveitando o mapa em cache para não redesenhar as geometrias.
- Garanta que nenhuma categoria desconhecida seja enviada ao modelo ao montar as linhas do município clicado.
- Modo 3: a função de predição em `model_utils.py` deve levantar um erro explícito (ex.: `ValueError`) se a combinação município × produto não estiver em `combinacoes_validas`, e a interface deve capturá-lo e exibir a mensagem de bloqueio.

# VERIFICAÇÃO NA INICIALIZAÇÃO
Implemente `validar_ambiente()` em `model_utils.py`, executada uma vez (`st.cache_resource`) antes de renderizar qualquer modo. Se uma checagem bloqueante falhar, mostre `st.error` com a causa e chame `st.stop()`.

Checagens bloqueantes:
1. `booster.feature_names` é igual, na mesma ordem, a `FEATURES` em `config.py`.
2. `feature_types` marca como categóricas exatamente: regiao, estado, cidade, produto, tipo_lavoura.
3. Cada lista de `categorias.json` não tem duplicatas e contém todos os valores da respectiva coluna na base.
4. Cada produto tem um único `tipo_lavoura` na base. Se não, liste os produtos ambíguos.
5. `ANO_REFERENCIA` em `config.py` é igual ao maior valor de `ano` na base (2024).
6. Teste de fumaça: amostra fixa (seed) de ~2.000 linhas da base, montada pela MESMA função de montagem usada nos modos (nesta checagem, use o `ano` real de cada linha). Todas as predições devem ser finitas e, após o `expm1`, >= 0.

Aviso (não bloqueante): se o Spearman global entre `rendimento_ha` real e previsto nessa amostra for < 0,8 (limiar em `config.py`, ajustável), mostre `st.warning` de possível divergência de categorias ou de ordem. Como a amostra vem do treino, esse teste só detecta erro de montagem e não mede a qualidade do modelo.

Se a versão do xgboost gravar as categorias dentro do modelo, compare-as também com `categorias.json`.

# LIMITAÇÕES A EXIBIR NA INTERFACE
- Os resultados são rendimento previsto por hectare, e não volume de produção. A base não contém área plantada ou colhida, então o app não estima quanto o município produz no total.
- As estimativas assumem o nível tecnológico e as práticas do período mais recente da base (2024). O modelo não projeta crescimento futuro.
- Use os resultados para comparar municípios e produtos entre si; o valor absoluto previsto tem margem de erro relevante. O app é uma ferramenta de ranqueamento relativo, e não de previsão precisa.
- Na agregação por região ou estado (Modo 2), todos os municípios têm o mesmo peso (mediana simples), independentemente do tamanho da área cultivada.
- A predição é uma estimativa do modelo, não garantia de produtividade.

# BOAS PRÁTICAS
- Estrutura modular: `app.py`, `model_utils.py` (carga, montagem, predição e validação), `geo_utils.py`, `index_utils.py`, `config.py`.
- Docstrings e type hints. Nenhuma lógica de treino no app.

# ENTREGÁVEIS
1. Estrutura de pastas.
2. Código completo de cada arquivo.
3. `build_reference.py` gerando as tabelas auxiliares e `categorias.json`. Ele usa a base única e os dados do IBGE baixados pelas URLs acima, NÃO aplica filtros nem tratamento de outliers, calcula `referencia_produto` em escala original (sem log) e `clima_base` sobre os últimos 10 anos, e imprime um relatório (linhas da base, produtos e municípios, não casados nos dois sentidos, municípios com `clima_base_fallback`, contagens da validação da base).
4. `model_utils.py` deve deixar explícito o ponto único onde o `expm1` é aplicado.
5. `requirements.txt` com versões fixadas (xgboost 3.4.1, a mesma do treino).
6. Instruções para rodar (`python build_reference.py` e depois `streamlit run app.py`).
7. Lista de suposições e pontos a confirmar.

# CRITÉRIOS DE ACEITE
- O app abre e carrega o modelo sem erros.
- A verificação de inicialização passa com o modelo e a base fornecidos. Ao forçar uma divergência em teste (ex.: ordem de features trocada), o app aborta com mensagem clara.
- Nenhum modo exibe seletor de ano, e toda predição do app usa `ano = ANO_REFERENCIA` (verificável por teste que inspeciona o DataFrame enviado ao `predict`).
- O mapa renderiza todos os municípios em menos de ~5 s após a ação.
- Com o usuário em outro modo, a predição do Modo 1 não é executada.
- Nenhum ranking ou gráfico compara valores brutos de produtos com unidades diferentes.
- Alterar clima ou produto muda os resultados de forma coerente.
- No Modo 2, clicar em um município atualiza o painel à direita em menos de ~2 s, sem perder o zoom do mapa.
- O município clicado corresponde ao `codigo_ibge` correto (teste com ao menos um nome de cidade repetido em dois estados).
- No Modo 3, o selectbox de produto lista apenas produtos válidos para o município escolhido, e a troca de município atualiza a lista.
- No Modo 3, uma combinação município × produto fora de `combinacoes_validas` (testada forçando o valor via código) é bloqueada, sem chamar o `predict`, e exibe a mensagem de bloqueio.
- No Modo 3, o botão "Restaurar clima base" funciona sem erro e restaura os 6 campos com o clima base dos últimos 10 anos.
