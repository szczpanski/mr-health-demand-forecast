
from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from .data_preprocessing import carregar_dados, validar_dados, consolidar_base, criar_demanda_diaria
from .feature_engineering import criar_features
from .train import modelos_candidatos, FEATURES
from .evaluate import avaliar_modelos, metricas
from .predict import prever_futuro


def executar(pedido, item_pedido, itens, saida="outputs") -> dict:
    saida = Path(saida)
    (saida / "metrics").mkdir(parents=True, exist_ok=True)
    (saida / "models").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("data/predictions").mkdir(parents=True, exist_ok=True)

    df_pedido, df_item_pedido, df_itens = carregar_dados(pedido, item_pedido, itens)
    qualidade = validar_dados(df_pedido, df_item_pedido, df_itens)
    base, pedidos = consolidar_base(df_pedido, df_item_pedido, df_itens)
    demanda = criar_demanda_diaria(base)
    features = criar_features(demanda, df_itens)

    modelos = modelos_candidatos()
    detalhado, resumo = avaliar_modelos(features, modelos)
    resumo.to_csv(saida / "metrics" / "comparacao_modelos.csv", index=False)
    detalhado.to_csv(saida / "metrics" / "validacao_temporal.csv", index=False)

    melhor_nome = resumo.iloc[0]["modelo"]
    modelos_ml = set(modelos)
    if melhor_nome not in modelos_ml:
        # Mantém o melhor modelo de ML para persistência e previsão futura,
        # mas registra honestamente quando um baseline vence a validação.
        melhor_ml = resumo[resumo["modelo"].isin(modelos_ml)].iloc[0]["modelo"]
    else:
        melhor_ml = melhor_nome

    datas = sorted(features["DATA_PEDIDO"].unique())
    corte = datas[-14]
    treino = features[features["DATA_PEDIDO"] < corte]
    teste = features[features["DATA_PEDIDO"] >= corte]
    modelo = modelos[melhor_ml]
    modelo.fit(treino[FEATURES], treino["QUANTIDADE"])
    pred = np.clip(modelo.predict(teste[FEATURES]), 0, None)
    holdout = metricas(teste["QUANTIDADE"], pred)
    holdout.update({"modelo_ml": melhor_ml, "vencedor_validacao": melhor_nome})

    joblib.dump(modelo, saida / "models" / "best_model.joblib")
    previsao = prever_futuro(modelo, demanda, df_itens, horizonte=7)
    previsao.to_csv("data/predictions/previsao_7_dias.csv", index=False)
    base.to_csv("data/processed/vendas_consolidadas.csv", index=False)
    demanda.to_csv("data/processed/demanda_diaria.csv", index=False)

    payload = {"qualidade": qualidade, "holdout": holdout}
    (saida / "metrics" / "resumo_execucao.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pedido", default="data/raw/PEDIDO-_1_.xlsx")
    parser.add_argument("--item-pedido", default="data/raw/ITEM_PEDIDO-_2_.xlsx")
    parser.add_argument("--itens", default="data/raw/ITENS-_3_.xlsx")
    parser.add_argument("--saida", default="outputs")
    args = parser.parse_args()
    print(executar(args.pedido, args.item_pedido, args.itens, args.saida))


if __name__ == "__main__":
    main()
