
from __future__ import annotations

from pathlib import Path
from typing import Tuple
import pandas as pd


def normalizar_nome_coluna(valor: object) -> str:
    """Padroniza nomes de colunas para letras maiúsculas e underscore."""
    return str(valor).strip().upper().replace(" ", "_")


def carregar_dados(
    caminho_pedido: str | Path,
    caminho_item_pedido: str | Path,
    caminho_itens: str | Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carrega e padroniza as três planilhas do case."""
    pedido = pd.read_excel(caminho_pedido)
    item_pedido = pd.read_excel(caminho_item_pedido)
    itens = pd.read_excel(caminho_itens)

    for df in (pedido, item_pedido, itens):
        df.columns = [normalizar_nome_coluna(c) for c in df.columns]

    pedido = pedido.loc[:, ~pedido.columns.str.startswith("UNNAMED")].copy()
    item_pedido = item_pedido.loc[:, ~item_pedido.columns.str.startswith("UNNAMED")].copy()

    if itens.shape[1] != 2:
        raise ValueError(f"ITENS deveria possuir duas colunas; foram encontradas {itens.shape[1]}.")

    itens.columns = ["ID_ITEM", "VALOR_UNITARIO"]
    pedido["DATA_PEDIDO"] = pd.to_datetime(pedido["DATA_PEDIDO"], errors="coerce")
    pedido["ID_PEDIDO"] = pd.to_numeric(pedido["ID_PEDIDO"], errors="coerce")
    item_pedido["ID_PEDIDO"] = pd.to_numeric(item_pedido["ID_PEDIDO"], errors="coerce")
    item_pedido["ID_ITEM"] = item_pedido["ID_ITEM"].astype(str).str.strip().str.upper()
    item_pedido["QUANTIDADE"] = pd.to_numeric(item_pedido["QUANTIDADE"], errors="coerce")
    itens["ID_ITEM"] = itens["ID_ITEM"].astype(str).str.strip().str.upper()
    itens["VALOR_UNITARIO"] = pd.to_numeric(itens["VALOR_UNITARIO"], errors="coerce")

    return pedido, item_pedido, itens


def validar_dados(
    pedido: pd.DataFrame,
    item_pedido: pd.DataFrame,
    itens: pd.DataFrame,
) -> dict:
    """Retorna um resumo de qualidade e falha em problemas críticos."""
    resumo = {
        "pedidos": int(len(pedido)),
        "linhas_item_pedido": int(len(item_pedido)),
        "itens": int(len(itens)),
        "datas_invalidas": int(pedido["DATA_PEDIDO"].isna().sum()),
        "quantidades_invalidas": int(item_pedido["QUANTIDADE"].isna().sum()),
        "precos_invalidos": int(itens["VALOR_UNITARIO"].isna().sum()),
        "pedidos_sem_itens": int((~pedido["ID_PEDIDO"].isin(item_pedido["ID_PEDIDO"])).sum()),
        "itens_sem_cadastro": int((~item_pedido["ID_ITEM"].isin(itens["ID_ITEM"])).sum()),
    }
    criticos = ["datas_invalidas", "quantidades_invalidas", "precos_invalidos",
                "pedidos_sem_itens", "itens_sem_cadastro"]
    erros = {k: resumo[k] for k in criticos if resumo[k] > 0}
    if erros:
        raise ValueError(f"Problemas críticos de qualidade: {erros}")
    return resumo


def consolidar_base(
    pedido: pd.DataFrame,
    item_pedido: pd.DataFrame,
    itens: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Consolida linhas repetidas e reconstrói receita por pedido."""
    itens_pedido_agg = (
        item_pedido.groupby(["ID_PEDIDO", "ID_ITEM"], as_index=False)["QUANTIDADE"].sum()
    )
    base = (
        itens_pedido_agg
        .merge(pedido[["ID_PEDIDO", "DATA_PEDIDO"]], on="ID_PEDIDO", how="inner", validate="many_to_one")
        .merge(itens, on="ID_ITEM", how="left", validate="many_to_one")
    )
    base["RECEITA_ITEM"] = base["QUANTIDADE"] * base["VALOR_UNITARIO"]
    total_pedido = (
        base.groupby("ID_PEDIDO", as_index=False)["RECEITA_ITEM"]
        .sum().rename(columns={"RECEITA_ITEM": "VALOR_TOTAL_RECONSTRUIDO"})
    )
    pedido_tratado = pedido.drop(columns=["VALOR_TOTAL"], errors="ignore").merge(
        total_pedido, on="ID_PEDIDO", how="left"
    )
    return base, pedido_tratado


def criar_demanda_diaria(base: pd.DataFrame) -> pd.DataFrame:
    """Cria painel completo data x produto, incluindo dias sem venda."""
    vendas = (
        base.groupby(["DATA_PEDIDO", "ID_ITEM"], as_index=False)
        .agg(QUANTIDADE=("QUANTIDADE", "sum"), RECEITA=("RECEITA_ITEM", "sum"))
    )
    datas = pd.date_range(vendas["DATA_PEDIDO"].min(), vendas["DATA_PEDIDO"].max(), freq="D")
    produtos = sorted(vendas["ID_ITEM"].unique())
    grade = pd.MultiIndex.from_product([datas, produtos], names=["DATA_PEDIDO", "ID_ITEM"]).to_frame(index=False)
    demanda = grade.merge(vendas, on=["DATA_PEDIDO", "ID_ITEM"], how="left")
    demanda[["QUANTIDADE", "RECEITA"]] = demanda[["QUANTIDADE", "RECEITA"]].fillna(0)
    return demanda.sort_values(["ID_ITEM", "DATA_PEDIDO"]).reset_index(drop=True)
