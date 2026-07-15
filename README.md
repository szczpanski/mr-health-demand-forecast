# Mr. Health — Case técnico

Projeto de Ciência de Dados para previsão diária de demanda por produto, criado para apoiar decisões de estoque em uma rede de alimentação saudável.

## Problema de negócio

A empresa consolida manualmente planilhas de pedidos e deseja automatizar o processo, identificar padrões de venda e utilizar previsões para reduzir rupturas e desperdícios.

```
## Estrutura

```text
mr-health-demand-forecast/
├── data/
│   ├── raw             # planilhas xlsx
├── notebooks/
│   └── demand_forecasting.ipynb
├── src
├── outputs/
│   ├── figures
├── run_pipeline.py
├── requirements.txt
└── README.md
```

## Tecnologias

Python, Pandas, NumPy, scikit-learn, XGBoost, Matplotlib, Jupyter, Joblib e GitHub Actions.

## Como executar

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

Abra `notebooks/demand_forecasting.ipynb`, faça upload das três planilhas quando solicitado e execute as células em ordem.

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
