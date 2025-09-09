import joblib
import numpy as np


def main():
    model = joblib.load(
        "/home/enzo/Programming/ia-monitoramento-geladeira/treinamento/modelo_ia.pkl"
    )

    dado_teste = np.array([[20, 15, 5]])  # TempS1;TempS2;Diff;

    prediction = model.predict(dado_teste)

    print(f"Retorno da IA: {prediction[0]}")


if __name__ == "__main__":
    main()
