import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

df = pd.read_csv("data.csv")

colunas_independentes_x = ["Categoria", "N°Passageiros", "Cap.PortaMalas", "ArCondicionado", "Câmbio"]
colunas_dependentes_y = ["Valor"]

dados_x = df[colunas_independentes_x]
dados_y = df[colunas_dependentes_y]

modelo = LinearRegression().fit(dados_x, dados_y)

num_categoria_test = 1
num_npassageiros_test = 5
num_capportamalas_test = 2
num_arcondicionado_test = 1
num_cambio_test = 0

valores_test = np.array([[num_categoria_test, num_npassageiros_test, num_capportamalas_test, num_arcondicionado_test, num_cambio_test]])

predicao = modelo.predict(valores_test)

print(predicao)