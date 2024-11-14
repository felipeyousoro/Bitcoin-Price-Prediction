import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score


class RegressorValidator:
    def __init__(self, model):
        """
        Inicializa o validador com o modelo de regressão fornecido.
        """
        self.model = model

    def train(self, X_train, y_train):
        """
        Treina o modelo com os dados de treino.
        """
        self.model.fit(X_train, y_train)

    def validate(self, X_test, y_test):
        """
        Valida o modelo com os dados de teste e calcula as métricas de erro.

        Retorna o erro quadrático médio (MSE), o erro absoluto médio (MAE),
        o coeficiente de determinação (R²), o erro percentual absoluto médio (MAPE)
        e as previsões do modelo.
        """
        predictions = self.model.predict(X_test)
        print(predictions, y_test)

        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

        return mse, mae, r2, mape, predictions
