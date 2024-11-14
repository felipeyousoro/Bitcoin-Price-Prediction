# Bitcoin Price Prediction

Este trabalho é uma aplicação de algorotimos de aprendizado de máquina para a previsão do preço do Bitcoin

## Conjunto de dados

O dataset utilizado foi obtido no site [Kaggle](https://www.kaggle.com/mczielinski/bitcoin-historical-data) e contém dados históricos do preço do Bitcoin desde 2012. Os dados originalmente estão separados por minuto, neste trabalho foi realizado um processamento para agrupar os dados por hora

Extraia os dados e coloque-os na pasta `Projeto/dataset`

## Dependências

Todas as dependências do projeto estão listadas no arquivo `requirements.txt`. Para instalar as dependências, execute o comando:

```bash
cd Projeto
pip install -r requirements.txt
```

## Iniciar

Para executar o projeto, é necessário ter o Python instalado. Recomenda-se a utilização de um ambiente virtual para instalar as dependências do projeto. Para criar um ambiente virtual, execute o comando:

```bash
cd Projeto
streamlit run __main__.py
```