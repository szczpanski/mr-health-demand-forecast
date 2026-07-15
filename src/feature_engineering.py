
from __future__ import annotations
import numpy as np
import pandas as pd


def criar_features(demanda: pd.DataFrame, itens: pd.DataFrame) -> pd.DataFrame:
    """Cria atributos de calendário, tendência e defasagens sem vazamento temporal."""
    df = demanda.copy().sort_values(["ID_ITEM", "DATA_PEDIDO"])
    df = df.merge(itens, on="ID_ITEM", how="left")
    inicio = df["DATA_PEDIDO"].min()

    df["DIA_SEMANA"] = df["DATA_PEDIDO"].dt.dayofweek
    df["NOME_DIA"] = df["DATA_PEDIDO"].dt.day_name()
    df["FIM_SEMANA"] = (df["DIA_SEMANA"] >= 5).astype(int)
    df["MES"] = df["DATA_PEDIDO"].dt.month
    df["DIA_MES"] = df["DATA_PEDIDO"].dt.day
    df["SEMANA_ANO"] = df["DATA_PEDIDO"].dt.isocalendar().week.astype(int)
    df["TENDENCIA"] = (df["DATA_PEDIDO"] - inicio).dt.days

    grupo = df.groupby("ID_ITEM", group_keys=False)["QUANTIDADE"]
    df["LAG_1"] = grupo.shift(1)
    df["LAG_7"] = grupo.shift(7)
    df["LAG_14"] = grupo.shift(14)
    df["MEDIA_MOVEL_7"] = grupo.transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    df["MEDIA_MOVEL_14"] = grupo.transform(lambda s: s.shift(1).rolling(14, min_periods=1).mean())
    df["DESVIO_MOVEL_7"] = grupo.transform(lambda s: s.shift(1).rolling(7, min_periods=2).std())
    df["MIN_MOVEL_7"] = grupo.transform(lambda s: s.shift(1).rolling(7, min_periods=1).min())
    df["MAX_MOVEL_7"] = grupo.transform(lambda s: s.shift(1).rolling(7, min_periods=1).max())
    return df.replace([np.inf, -np.inf], np.nan)
