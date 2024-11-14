from datetime import datetime

import streamlit as st

from DataManager import DataManager


def st_data_visualization():
    st.title("Visualizar Estrutura dos Dados")

    data = DataManager.load_data()

    index_datetime = data.index.to_pydatetime()
    min_date, max_date = index_datetime[0], index_datetime[-1]
    default_start_date = datetime(2023, 1, 1)

    st.write("Selecione o intervalo de datas:")
    start_date, end_date = st.slider(
        "Intervalo para visualização",
        min_value=min_date,
        max_value=max_date,
        value=(default_start_date, max_date),
        format="MM/DD/YYYY"
    )

    st.write(f"**Intervalo selecionado:** {start_date.strftime('%d/%m/%Y')} a {end_date.strftime('%d/%m/%Y')}")

    filtered_data = data.loc[start_date:end_date]

    st.subheader("📉 Gráfico de Preço de Fechamento")
    st.line_chart(filtered_data['Close'])

    with st.expander("👀 Estrutura do Dataset"):
        st.write(filtered_data.head())

    with st.expander("📊 Estatísticas Descritivas"):
        st.write(filtered_data.describe())

    with st.expander("📝 Descrição das Colunas"):
        st.write("""
        **Colunas Originais do Dataset:**
        - 🟢 **Open**: Preço de abertura. ⚠️ Esta coluna não será utilizada no modelo, pois a informação não está disponível no momento da previsão.
        - 🟢 **High**: Preço mais alto. ⚠️ Esta coluna não será utilizada no modelo, pois a informação não está disponível no momento da previsão.
        - 🟢 **Low**: Preço mais baixo. ⚠️ Esta coluna não será utilizada no modelo, pois a informação não está disponível no momento da previsão.
        - 🟢 **Volume**: Volume de negociações. ⚠️ Esta coluna não será utilizada no modelo, pois a informação não está disponível no momento da previsão.
        - 🟢 **Close**: Preço de fechamento. 🎯 A variável alvo do modelo de previsão.

        **Colunas Adicionadas a Partir das Transformações dos Dados:**
        - 🔵 **Hour**: Hora do dia (0-23).
        - 🔵 **Day**: Dia do mês (1-31).
        - 🔵 **Month**: Mês do ano (1-12).
        - 🔵 **Year**: Ano de observação.
        - 🔵 **DayOfWeek**: Dia da semana (0 = segunda-feira, 6 = domingo).
        - 🔵 **Close_lag_1d**: Preço de fechamento do dia anterior.
        - 🔵 **Close_lag_3d**: Preço de fechamento de três dias atrás.
        - 🔵 **Close_lag_7d**: Preço de fechamento de sete dias atrás.

        **Observações Importantes:**
        - As colunas com símbolo **🟢** indicam que são **colunas originais** do dataset, mas algumas não serão utilizadas no modelo.
        - As colunas com símbolo **🔵** indicam **colunas transformadas** a partir dos dados originais e serão utilizadas no modelo de previsão.
        """)

