import pandas as pd
import streamlit as st
from sklearn.svm import SVR

import utils
from DataManager import DataManager
from RegressorValidator import RegressorValidator


def st_select_svr_hyperparams():
    """
    Configura o seletor de hiperparâmetros do SVR no sidebar.
    """
    with st.sidebar.expander("Ajuste de hiperparâmetros", expanded=True):
        col1, col2 = st.columns(2)
        C = col1.number_input("C", 0.01, 1000.0, 18.37)
        epsilon = col2.number_input("epsilon", 0.01, 1.0, 0.31)

        kernel = st.radio("Kernel", ["linear", "poly", "rbf", "sigmoid"], index=1)

    return C, epsilon, kernel


def display_results(mse, mae, r2, mape):
    """
    Exibe as métricas de desempenho do modelo.
    """
    results = {
        "Métrica": ["Mean Squared Error (MSE)", "Mean Absolute Error (MAE)",
                    "R^2 Score", "Mean Absolute Percentage Error (MAPE)"],
        "Valor": [mse, mae, r2, mape]
    }
    results_df = pd.DataFrame(results)
    st.write("## Resultados")
    st.table(results_df)


def prepare_data(selected_features, start_date, end_date, steps):
    """
    Carrega, filtra e prepara os dados para treinamento e teste.
    """
    data = DataManager.load_data()
    filtered_data = data.loc[start_date:end_date]
    features_target = DataManager.prepare_features_and_target(filtered_data, selected_features)
    return DataManager.train_test_split(*features_target, steps=steps)


def st_svr():
    st.write("# Support Vector Machine (SVM) - Regressão")

    selected_features = utils.st_select_features()
    C, epsilon, kernel = st_select_svr_hyperparams()

    data = DataManager.load_data()
    start_date, end_date = utils.st_select_date_range(data)
    steps = st.number_input("Selecione a quantidade de passos (horas) a serem preditas", min_value=24, max_value=240, value=24, step=1)

    if st.button("Realizar Fit"):
        svr_model = SVR(C=C, epsilon=epsilon, kernel=kernel)
        validator = RegressorValidator(svr_model)

        # Preparar os dados
        X_train, X_test, y_train, y_test = prepare_data(selected_features, start_date, end_date, steps)

        # Treinar e validar o modelo
        validator.train(X_train, y_train)
        mse, mae, r2, mape, predictions = validator.validate(X_test, y_test)

        display_results(mse, mae, r2, mape)

        utils.st_plot_results(data, y_test, predictions, steps)
        utils.st_plot_scatter(y_test, predictions)
