from datetime import datetime

import matplotlib.pyplot as plt
import streamlit as st


def st_select_features():
    """
    Cria um conjunto de caixas de seleção no painel lateral do Streamlit para que o usuário selecione as
    features a serem usadas na análise.
    """
    with st.sidebar.expander("Selecione as features", expanded=False):
        col1, col2, col3 = st.columns(3)
        hour = col1.checkbox("Hour", True)
        day = col2.checkbox("Day", True)
        month = col3.checkbox("Month", True)

        col4, col5, _ = st.columns(3)
        year = col4.checkbox("Year", True)
        dayofweek = col5.checkbox("Day Of Week", True)

        col6, col7, col8 = st.columns(3)
        close_lag_1d = col6.checkbox("Close Lag 1D", True)
        close_lag_3d = col7.checkbox("Close Lag 3D", True)
        close_lag_7d = col8.checkbox("Close Lag 7D", True)

        col9, col10, col11 = st.columns(3)
        _open = col9.checkbox("Open", False)
        high = col10.checkbox("High", False)
        low = col11.checkbox("Low", False)

        volume = st.checkbox("Volume", False)

    return {
        'Hour': hour,
        'Day': day,
        'Month': month,
        'Year': year,
        'DayOfWeek': dayofweek,
        'Close_lag_1d': close_lag_1d,
        'Close_lag_3d': close_lag_3d,
        'Close_lag_7d': close_lag_7d,
        'Open': _open,
        'High': high,
        'Low': low,
        'Volume': volume
    }


def st_select_date_range(data):
    """
    Exibe um slider no Streamlit para o usuário selecionar um intervalo de datas com base nos dados
    fornecidos. O intervalo vai de `min_date` a `max_date`, com uma data inicial padrão de 1º de janeiro de 2023.
    """
    index_datetime = data.index.to_pydatetime()
    min_date, max_date = index_datetime[0], index_datetime[-1]
    default_start_date = datetime(2023, 1, 1)

    st.write("Selecione o intervalo de tempo para treinamento:")
    start_date = st.slider(
        "Intervalo para treinamento",
        min_value=min_date,
        max_value=max_date,
        value=default_start_date,
        format="MM/DD/YYYY"
    )

    # st.line_chart(data.loc[start_date:max_date]['Close'])

    return start_date, max_date


def st_plot_results(dataset, y_test, predictions, steps):
    """
    Plota os resultados comparando os preços reais e previstos de fechamento de acordo com o conjunto de dados
    fornecido, para um número determinado de passos/horas (steps).
    """
    plotting_data = dataset[-7 * steps:]

    plt.figure(figsize=(14, 7))
    plt.plot(plotting_data.index, plotting_data['Close'], label="Preço de Fechamento", color='b')
    plt.plot(y_test.index, y_test, label="Preço Real", color='green')
    plt.plot(y_test.index, predictions, label="Preço Previsto", color='r')

    plt.xlabel("Data")
    plt.ylabel("Preço de Fechamento")
    plt.title("Preço de Fechamento Real vs. Previsto")
    plt.legend()

    st.pyplot(plt)


def st_plot_scatter(y_test, predictions):
    """
    Plota um gráfico de dispersão comparando as previsões e os valores reais.
    """
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, predictions, color='blue', edgecolor='white', alpha=0.7)

    # Linha de referência y=x para comparar previsões com valores reais
    min_val = min(y_test.min(), predictions.min())
    max_val = max(y_test.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

    plt.xlabel("Valores Reais")
    plt.ylabel("Previsões")
    plt.title("Comparação entre Previsões e Valores Reais")

    st.pyplot(plt)