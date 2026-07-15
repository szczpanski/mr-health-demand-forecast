
from __future__ import annotations
from typing import Dict
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42

CATEGORICAS = ["ID_ITEM", "DIA_SEMANA", "MES"]
NUMERICAS = [
    "VALOR_UNITARIO", "FIM_SEMANA", "DIA_MES", "SEMANA_ANO", "TENDENCIA",
    "LAG_1", "LAG_7", "LAG_14", "MEDIA_MOVEL_7", "MEDIA_MOVEL_14",
    "DESVIO_MOVEL_7", "MIN_MOVEL_7", "MAX_MOVEL_7",
]
FEATURES = CATEGORICAS + NUMERICAS


def _ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def construir_pipeline(modelo, escalar: bool = False) -> Pipeline:
    etapas_num = [("imputer", SimpleImputer(strategy="median"))]
    if escalar:
        etapas_num.append(("scaler", StandardScaler()))
    pre = ColumnTransformer([
        ("categoricas", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _ohe()),
        ]), CATEGORICAS),
        ("numericas", Pipeline(etapas_num), NUMERICAS),
    ])
    return Pipeline([("preprocessamento", pre), ("modelo", modelo)])


def modelos_candidatos() -> Dict[str, Pipeline]:
    return {
        "Regressão Linear": construir_pipeline(LinearRegression(), escalar=True),
        "Poisson": construir_pipeline(PoissonRegressor(alpha=1.0, max_iter=2000), escalar=True),
        "Random Forest": construir_pipeline(RandomForestRegressor(
            n_estimators=300, max_depth=5, min_samples_leaf=4,
            random_state=RANDOM_STATE, n_jobs=-1
        )),
        "Gradient Boosting": construir_pipeline(GradientBoostingRegressor(
            n_estimators=150, max_depth=2, learning_rate=0.03,
            loss="huber", random_state=RANDOM_STATE
        )),
    }
