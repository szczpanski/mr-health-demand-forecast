
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .train import FEATURES


def metricas(y_real, y_previsto) -> dict:
    y_real = np.asarray(y_real)
    y_previsto = np.clip(np.asarray(y_previsto), 0, None)
    mascara = y_real != 0
    mape = float(np.mean(np.abs((y_real[mascara] - y_previsto[mascara]) / y_real[mascara])) * 100) if mascara.any() else np.nan
    return {
        "MAE": float(mean_absolute_error(y_real, y_previsto)),
        "RMSE": float(np.sqrt(mean_squared_error(y_real, y_previsto))),
        "MAPE_%": mape,
        "R2": float(r2_score(y_real, y_previsto)),
    }


def criar_folds_temporais(df: pd.DataFrame, horizonte: int = 14, n_folds: int = 3):
    datas = np.array(sorted(df["DATA_PEDIDO"].unique()))
    folds = []
    for i in range(n_folds, 0, -1):
        fim = len(datas) - horizonte * (i - 1)
        inicio = fim - horizonte
        datas_val = datas[inicio:fim]
        datas_treino = datas[:inicio]
        if len(datas_treino) and len(datas_val):
            folds.append((datas_treino, datas_val))
    return folds


def avaliar_modelos(df: pd.DataFrame, modelos: dict, horizonte: int = 14, n_folds: int = 3):
    resultados = []
    for fold, (datas_treino, datas_val) in enumerate(criar_folds_temporais(df, horizonte, n_folds), 1):
        treino = df[df["DATA_PEDIDO"].isin(datas_treino)].copy()
        validacao = df[df["DATA_PEDIDO"].isin(datas_val)].copy()

        # Baselines
        media_item = treino.groupby("ID_ITEM")["QUANTIDADE"].mean()
        media_item_dia = treino.groupby(["ID_ITEM", "DIA_SEMANA"])["QUANTIDADE"].mean()
        preds = {
            "Baseline média por item": validacao["ID_ITEM"].map(media_item).fillna(treino["QUANTIDADE"].mean()),
            "Baseline item + dia": [
                media_item_dia.get((item, dia), media_item.get(item, treino["QUANTIDADE"].mean()))
                for item, dia in zip(validacao["ID_ITEM"], validacao["DIA_SEMANA"])
            ],
            "Baseline sazonal lag 7": validacao["LAG_7"].fillna(validacao["MEDIA_MOVEL_7"]).fillna(0),
        }
        for nome, pred in preds.items():
            resultados.append({"fold": fold, "modelo": nome, **metricas(validacao["QUANTIDADE"], pred)})

        for nome, pipeline in modelos.items():
            pipeline.fit(treino[FEATURES], treino["QUANTIDADE"])
            pred = np.clip(pipeline.predict(validacao[FEATURES]), 0, None)
            resultados.append({"fold": fold, "modelo": nome, **metricas(validacao["QUANTIDADE"], pred)})

    detalhado = pd.DataFrame(resultados)
    resumo = detalhado.groupby("modelo", as_index=False).agg(
        MAE_medio=("MAE", "mean"), RMSE_medio=("RMSE", "mean"),
        MAPE_medio=("MAPE_%", "mean"), R2_medio=("R2", "mean"),
    ).sort_values("MAE_medio")
    return detalhado, resumo
