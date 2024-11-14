import streamlit as st

from data_visualization import st_data_visualization
from random_forest import st_random_forest
from _xgboost import st_xgboost
from svm import st_svr

if __name__ == '__main__':
    pages = [
        "Estrutura Geral dos Dados",
        "Random Forest Regressor",
        "XGBoost Regressor",
        "Support Vector Machine (SVM)"
    ]

    st.sidebar.title("Painel de Navegação")

    if "page_index" not in st.session_state:
        st.session_state.page_index = 0

    col1, col2, col3, col4, col5, col6 = st.sidebar.columns(6)
    if col5.button("←") and st.session_state.page_index > 0:
        st.session_state.page_index -= 1
    if col6.button("→") and st.session_state.page_index < len(pages) - 1:
        st.session_state.page_index += 1

    page = st.sidebar.selectbox("Go to", pages, index=st.session_state.page_index)

    st.session_state.page_index = pages.index(page)
    st.sidebar.markdown("<hr style='border: 2px solid white;'>", unsafe_allow_html=True)

    if page == "Estrutura Geral dos Dados":
        st_data_visualization()
    elif page == "Random Forest Regressor":
        st_random_forest()
    elif page == "XGBoost Regressor":
        st_xgboost()
    elif page == "Support Vector Machine (SVM)":
        st_svr()
