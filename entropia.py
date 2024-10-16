# Importar las bibliotecas necesarias
import pandas as pd
import numpy as np

# Paso 1: Preparar los datos
# Leer los datos del archivo CSV
data = pd.read_csv("data.csv")
print(data.head(5))

# Verificar si aún hay valores faltantes
print("Valores faltantes por columna:")
print(data.isnull().sum())     

# Seleccionar las columnas relevantes para el análisis
data_personality = data['Personality']
data_interest = data['Interest']



# Paso 2: Determinar qué se desea conocer
# Queremos conocer la distribución de clases en ambas categorías:
# - Personality: Tipos de personalidad (e.g., ENFP, INTP).
# - Interest: Intereses (e.g., Arts, Technology).

# Paso 3: Identificar las clases
# Contar las frecuencias de cada clase en ambas categorías
frecuencias_personality = data_personality.value_counts()
frecuencias_interest = data_interest.value_counts()

# Mostrar las clases identificadas con sus frecuencias
print("Clases - Personality:\n", frecuencias_personality)
print("\nClases - Interest:\n", frecuencias_interest)

# Paso 4: Calcular las probabilidades de las clases
# Calcular la probabilidad de cada clase dividiendo su frecuencia por el total
probabilidades_personality = frecuencias_personality / len(data_personality)
probabilidades_interest = frecuencias_interest / len(data_interest)

# Mostrar las probabilidades calculadas para cada clase
print("\nProbabilidades - Personality:\n", probabilidades_personality)
print("\nProbabilidades - Interest:\n", probabilidades_interest)

# Paso 5: Calcular la entropía del sistema
# Definir una función para calcular la entropía basada en las probabilidades
def calcular_entropia(probabilidades):
    log_probabilidades = np.log2(probabilidades)
    productos = probabilidades * log_probabilidades
    entropia = -productos.sum()
    return productos, entropia

# Calcular la entropía para Personality e Interest
productos_personality, entropia_personality = calcular_entropia(probabilidades_personality)
productos_interest, entropia_interest = calcular_entropia(probabilidades_interest)

# Mostrar los resultados de entropía
print("\nEntropía - Personality:", entropia_personality)
print("Entropía - Interest:", entropia_interest)

# Crear DataFrames detallados para visualizar los cálculos paso a paso
productos_personality_df = pd.DataFrame({
    'Clase': frecuencias_personality.index,
    'Probabilidad': probabilidades_personality.values,
    'log2(Probabilidad)': np.log2(probabilidades_personality.values),
    'Producto': productos_personality.values
})

productos_interest_df = pd.DataFrame({
    'Clase': frecuencias_interest.index,
    'Probabilidad': probabilidades_interest.values,
    'log2(Probabilidad)': np.log2(probabilidades_interest.values),
    'Producto': productos_interest.values
})

# Mostrar los DataFrames con los cálculos detallados
print("\nCálculo de Entropía - Personality:")
print(productos_personality_df)

print("\nCálculo de Entropía - Interest:")
print(productos_interest_df)
