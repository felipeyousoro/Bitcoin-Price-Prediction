import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt
from xgboost import XGBRegressor

import utils
from DataManager import DataManager
from RegressorValidator import RegressorValidator


def st_select_hyperparams():
    """
    Configura o seletor de hiperparâmetros no sidebar
    """
    with st.sidebar.expander("Ajuste de hiperparâmetros", expanded=True):
        col1, col2 = st.columns(2)
        max_depth = col1.number_input("max_depth", 1, 100, 6)
        n_estimators = col2.number_input("n_estimators", 1, 1000, 137)

        learning_rate = st.number_input("learning_rate", 0.01, 1.0, 0.05)
    return max_depth, n_estimators, learning_rate



def display_feature_importance(model, feature_columns):
    """
    Mostra a importância das features do modelo treinado
    """
    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": feature_importances
    })
    st.write("## Importância das Features")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis', ax=ax)
    ax.set_title('Importância das Features')
    ax.set_xlabel('Importância')
    ax.set_ylabel('Feature')
    st.pyplot(fig)


def display_results(mse, mae, r2, mape):
    """
    Exibe os resultados das métricas do modelo
    """
    results = {
        "Métrica": ["MSE (Erro Médio Quadrático)", "MAE (Erro Médio Absoluto)",
                    "R² (Coeficiente de Determinação)", "MAPE (Erro Médio Percentual Absoluto)"],
        "Valor": [mse, mae, r2, mape]
    }
    results_df = pd.DataFrame(results)
    st.write("## Resultados")
    st.table(results_df)


def st_xgboost():
    st.write("# XGBoost - Regressão")

    selected_features = utils.st_select_features()
    max_depth, n_estimators, learning_rate = st_select_hyperparams()

    data = DataManager.load_data()
    start_date, end_date = utils.st_select_date_range(data)

    steps = st.number_input("Selecione a quantidade de passos (horas) para predição", min_value=24, max_value=240,
                            value=24, step=1)

    if st.button("Executar treinamento"):
        model = XGBRegressor(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=42
        )

        # Preparar dados para treinamento
        filtered_data = data.loc[start_date:end_date]
        features_target = DataManager.prepare_features_and_target(filtered_data, selected_features)
        X_train, X_test, y_train, y_test = DataManager.train_test_split(*features_target, steps=steps)

        # Treinar e validar modelo
        validator = RegressorValidator(model)
        validator.train(X_train, y_train)
        mse, mae, r2, mape, predictions = validator.validate(X_test, y_test)

        display_results(mse, mae, r2, mape)

        display_feature_importance(model, X_train.columns)

        utils.st_plot_results(filtered_data, y_test, predictions, steps)
        utils.st_plot_scatter(y_test, predictions)
