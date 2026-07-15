
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def grafico_real_vs_previsto(resultado: pd.DataFrame, caminho: str | Path) -> None:
    diario = resultado.groupby("DATA_PEDIDO", as_index=False)[["REAL", "PREVISTO"]].sum()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(diario["DATA_PEDIDO"], diario["REAL"], marker="o", label="Real")
    ax.plot(diario["DATA_PEDIDO"], diario["PREVISTO"], marker="o", label="Previsto")
    ax.set_title("Demanda real x prevista")
    ax.set_xlabel("Data")
    ax.set_ylabel("Quantidade")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(caminho, dpi=160)
    plt.close(fig)
