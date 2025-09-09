import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib


def treinamento():
    df = pd.read_csv(
        "/home/enzo/Programming/ia-monitoramento-geladeira/treinamento/base.csv",
        delimiter=";",
        decimal=",",
    )

    print(f"O Dataset tem {df.shape[0]} linhas e {df.shape[1]} colunas\n")

    df = df.astype(
        {
            "TempS1": float,
            "UmiS1": float,
            "TempS2": float,
            "UmiS2": float,
            "Diff": float,
        }
    )

    # df["DtGravacao"] = pd.to_datetime(df["DtGravacao"], format="%Y-%m-%d %H:%M:%S")

    # df["DtGravacao"] = df["DtGravacao"].apply(lambda x: int(x.timestamp()))

    # VER com o Biel quantas vezes + ou - ele gerou erros na base de dados para preencher um contamination melhor:
    model = IsolationForest(contamination=0.5, random_state=42)

    model.fit(df[["TempS1", "TempS2", "Diff"]])

    df["Falha"] = model.predict(df[["TempS1", "TempS2", "Diff"]])

    linhas_falha = df["Falha"] == -1
    linhas_sem_falhas = df["Falha"] == 1

    print(f"Quantidade de linhas sem falhas: {df[linhas_sem_falhas].shape[0]} linhas\n")

    print(f"Quantidade de linhas com falhas: {df[linhas_falha].shape[0]} linhas\n")

    print(f"Exemplo 5 primeiras linhas com falhas: \n")

    print(df[linhas_falha].head(5), "\n")

    print(f"Exemplo 5 últimas linhas com falhas: \n")

    print(df[linhas_falha].tail(5), "\n")

    print("Verificando se diferença no range da temperatura geralmente da Falha")

    print(df[(df["Diff"] >= 0.8) & (df["Falha"] == -1)], "\n")
    print(df[(df["Diff"] >= 0.8) & (df["Falha"] == 1)], "\n")

    joblib.dump(model, "modelo_ia.pkl")


if __name__ == "__main__":
    treinamento()
