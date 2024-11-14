import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class DataManager():
    DATA_PATH = "dataset/btcusd_hourly_data2.csv"

    @classmethod
    def load_data(cls):
        """
        Carrega os dados do arquivo CSV, converte o campo 'Timestamp' para datetime,
        define-o como índice e cria colunas de características adicionais, incluindo
        lags para a coluna 'Close'
        """

        dataset = pd.read_csv(cls.DATA_PATH)
        dataset['Timestamp'] = pd.to_datetime(dataset['Timestamp'], unit='s')
        dataset.set_index('Timestamp', inplace=True)
        dataset.dropna(inplace=True)

        dataset['Hour'] = dataset.index.hour
        dataset['Day'] = dataset.index.day
        dataset['Month'] = dataset.index.month
        dataset['Year'] = dataset.index.year
        dataset['DayOfWeek'] = dataset.index.dayofweek
        dataset['Close_lag_1d'] = dataset['Close'].shift(24)
        dataset['Close_lag_3d'] = dataset['Close'].shift(3 * 24)
        dataset['Close_lag_7d'] = dataset['Close'].shift(7 * 24)

        return dataset

    @classmethod
    def scale_data(cls, dataset):
        """
        Normaliza as colunas numéricas do conjunto de dados para um intervalo de 0 a 1
        usando MinMaxScaler
        """

        scaler = MinMaxScaler()
        numeric_columns = dataset.select_dtypes(include=['float64', 'int64']).columns
        dataset[numeric_columns] = scaler.fit_transform(dataset[numeric_columns])

        return dataset

    @classmethod
    def prepare_features_and_target(cls, dataset, selected_features):
        """
        Seleciona as colunas de recursos (features) especificadas no dicionário
        'selected_features' e define a coluna 'Close' como alvo (target)
        """

        features = dataset[[feature for feature, selected in selected_features.items() if selected]]
        target = dataset['Close']

        return features, target

    @classmethod
    def train_test_split(cls, features, target, steps):
        """
        Divide os dados em conjuntos de treino e teste com base em uma quantidade
        de etapas (steps) especificada
        """

        X_train, X_test = features[:-steps], features[-steps:]
        y_train, y_test = target[:-steps], target[-steps:]

        return X_train, X_test, y_train, y_test
