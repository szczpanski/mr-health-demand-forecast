# 📈 Mr. Health — Demand Forecasting for Inventory Optimization

Projeto de Ciência de Dados para previsão diária de demanda por produto, criado para apoiar decisões de estoque em uma rede de alimentação saudável.

## Problema de negócio

A empresa consolida manualmente planilhas de pedidos e deseja automatizar o processo, identificar padrões de venda e utilizar previsões para reduzir rupturas e desperdícios.

A base não possui identificador de loja, estoque atual, ruptura, promoções, validade ou lead time. Portanto, o escopo tecnicamente possível é a **previsão global por produto**, não por unidade.

## Principais resultados

| Indicador | Resultado |
|---|---:|
| Pedidos | 181 |
| Produtos | 4 |
| Período | 93 dias |
| Quantidade vendida | 895 unidades |
| Receita reconstruída | R$ 15.960,00 |
| Ticket médio | R$ 88,18 |
| Produto de maior receita | Item D |
| Participação do item D | aproximadamente 39,3% |
| Melhor MAE de ML na validação | aproximadamente 2,21 unidades |
| MAE de ML no holdout | aproximadamente 1,82 unidade |

Um baseline simples apresentou desempenho competitivo no holdout. Isso é tratado como um resultado válido: com apenas três meses de histórico, um modelo complexo ainda não demonstra superioridade consistente.

## Insights de negócio

- A demanda aumenta fortemente entre sexta-feira e domingo.
- Domingo apresenta demanda média aproximadamente três vezes maior que segunda-feira.
- O item D merece prioridade no planejamento por concentrar a maior parcela da receita.
- A empresa deve ampliar o histórico para pelo menos 12–24 meses.
- Estoque, ruptura, desperdício, validade, promoções, feriados, canal e lead time devem ser incorporados em uma próxima versão.

## Arquitetura

```mermaid
flowchart LR
    A[Planilhas brutas] --> B[Validação e limpeza]
    B --> C[Consolidação de pedidos]
    C --> D[Painel diário produto x data]
    D --> E[EDA e atributos temporais]
    E --> F[Validação temporal]
    F --> G[Comparação com baselines]
    G --> H[Modelo persistido]
    H --> I[Previsão de 7 dias]
    I --> J[Recomendação de estoque]
```

## Estrutura

```text
mr-health-demand-forecast/
├── data/
│   ├── raw/              # planilhas locais, ignoradas pelo Git
│   ├── processed/        # dados tratados
│   └── predictions/      # previsões
├── notebooks/
│   └── 01_demand_forecasting.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── visualization.py
│   └── pipeline.py
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── models/
├── tests/
├── run_pipeline.py
├── requirements.txt
├── Makefile
└── README.md
```

## Tecnologias

Python, Pandas, NumPy, scikit-learn, XGBoost, Matplotlib, Jupyter, Joblib e GitHub Actions.

## Como executar localmente

```bash
git clone <URL-DO-SEU-REPOSITORIO>
cd mr-health-demand-forecast

python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

Coloque as planilhas em `data/raw/`:

```text
PEDIDO-_1_.xlsx
ITEM_PEDIDO-_2_.xlsx
ITENS-_3_.xlsx
```

Execute:

```bash
python run_pipeline.py
```

## Google Colab

Abra `notebooks/01_demand_forecasting.ipynb`, faça upload das três planilhas quando solicitado e execute as células em ordem.

## Validação

A divisão é temporal, sem embaralhamento. São utilizadas janelas sucessivas de validação e um holdout final com os últimos 14 dias.

Modelos avaliados:

- Regressão Linear
- Regressão de Poisson
- Random Forest
- Gradient Boosting
- XGBoost no notebook, quando disponível
- Baseline de média por produto
- Baseline por produto e dia da semana
- Baseline sazonal de sete dias

Métricas:

- MAE
- RMSE
- MAPE
- R²

## Persistência

Após a execução modular, o melhor modelo de Machine Learning é salvo em:

```text
outputs/models/best_model.joblib
```

A escolha é separada do vencedor geral da validação. Caso um baseline seja superior, isso é registrado de maneira transparente.

## Visualizações

![01_demanda_diaria](outputs/figures/01_demanda_diaria.png)

![02_demanda_por_item](outputs/figures/02_demanda_por_item.png)

![03_demanda_dia_semana](outputs/figures/03_demanda_dia_semana.png)


## Próximos passos

1. Incorporar unidade/loja e hierarquia geográfica.
2. Adicionar estoque, ruptura, desperdício e prazo de validade.
3. Criar variáveis de promoção, feriado, clima e canal.
4. Estimar intervalos de previsão e estoque de segurança por nível de serviço.
5. Implementar rastreamento de experimentos e monitoramento de erro em produção.

## Aviso sobre os dados

As planilhas originais não foram incluídas no repositório público. Verifique os termos do processo seletivo antes de publicar dados ou materiais fornecidos pela empresa.
