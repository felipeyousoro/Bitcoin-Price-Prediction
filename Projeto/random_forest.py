import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor

import utils
from DataManager import DataManager
from RegressorValidator import RegressorValidator


def st_select_hyperparams():
    with st.sidebar.expander("Ajuste de hiperparâmetros", expanded=True):
        col1, col2 = st.columns(2)
        max_depth = col1.number_input("max_depth", 1, 100, 3)
        n_estimators = col2.number_input("n_estimators", 1, 1000, 189)

        col3, col4 = st.columns(2)
        min_samples_split = col3.number_input("min_samples_split", 1, 100, 2)
        min_samples_leaf = col4.number_input("min_samples_leaf", 1, 100, 10)

        col5 = st.columns(1)[0]
        max_features = col5.number_input("max_features", 0.1, 1.0, 1.0)

    return max_depth, n_estimators, min_samples_split, min_samples_leaf, max_features


def st_random_forest():
    st.write("# Random Forest")

    selected_features = utils.st_select_features()
    max_depth, n_estimators, min_samples_split, min_samples_leaf, max_features = st_select_hyperparams()

    data = DataManager.load_data()
    start_date, end_date = utils.st_select_date_range(data)

    steps = st.number_input("Selecione a quantidade de passos (horas) a serem preditas", min_value=24, max_value=240,
                            value=24, step=1)

    if st.button("Realizar Fit"):
        rf = RandomForestRegressor(
            max_depth=max_depth,
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42
        )

        validation = RegressorValidator(rf)

        filtered_data = DataManager.load_data()
        filtered_data = filtered_data.loc[start_date:end_date]
        # filtered_data = DataManager.scale_data(filtered_data)
        features_target = DataManager.prepare_features_and_target(filtered_data, selected_features)
        X_train, X_test, y_train, y_test = DataManager.train_test_split(*features_target, steps=steps)

        validation.train(X_train, y_train)
        mse, mae, r2, mape, predictions = validation.validate(X_test, y_test)

        results = {
            "Métrica": ["Mean Squared Error (MSE)", "Mean Absolute Error (MAE)", "R^2 Score",
                        "Mean Absolute Percentage Error (MAPE)"],
            "Valor": [mse, mae, r2, mape]
        }
        results_df = pd.DataFrame(results)
        st.write("## Resultados")
        st.table(results_df)

        feature_importances = rf.feature_importances_
        importance_df = pd.DataFrame({
            "Feature": X_train.columns,
            "Importance": feature_importances
        })

        st.write("## Importância das Features")
        fig=plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
        plt.title('Feature Importances from Random Forest')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        st.pyplot(fig)

        utils.st_plot_results(filtered_data, y_test, predictions, steps)
        utils.st_plot_scatter(y_test, predictions)
