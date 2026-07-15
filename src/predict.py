
from __future__ import annotations
import numpy as np
import pandas as pd
from .feature_engineering import criar_features
from .train import FEATURES


def prever_futuro(modelo, demanda: pd.DataFrame, itens: pd.DataFrame, horizonte: int = 7) -> pd.DataFrame:
    """Previsão recursiva: cada dia previsto alimenta as defasagens seguintes."""
    historico = demanda.copy()
    produtos = sorted(historico["ID_ITEM"].unique())
    ultima_data = historico["DATA_PEDIDO"].max()
    saida = []

    for passo in range(1, horizonte + 1):
        data = ultima_data + pd.Timedelta(days=passo)
        novas = pd.DataFrame({"DATA_PEDIDO": data, "ID_ITEM": produtos, "QUANTIDADE": np.nan, "RECEITA": np.nan})
        temporario = pd.concat([historico, novas], ignore_index=True)
        feat = criar_features(temporario, itens)
        linhas = feat[feat["DATA_PEDIDO"].eq(data)].copy()
        previsoes = np.clip(modelo.predict(linhas[FEATURES]), 0, None)
        novas["QUANTIDADE"] = previsoes
        novas = novas.merge(itens, on="ID_ITEM", how="left")
        novas["RECEITA"] = novas["QUANTIDADE"] * novas["VALOR_UNITARIO"]
        saida.append(novas[["DATA_PEDIDO", "ID_ITEM", "QUANTIDADE", "RECEITA"]])
        historico = pd.concat([historico, novas[["DATA_PEDIDO", "ID_ITEM", "QUANTIDADE", "RECEITA"]]], ignore_index=True)

    return pd.concat(saida, ignore_index=True).rename(columns={"QUANTIDADE": "PREVISAO_DEMANDA"})
