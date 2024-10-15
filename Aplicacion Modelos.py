# Databricks notebook source
pip install pmdarima


# COMMAND ----------

pip install prophet

# COMMAND ----------

import pandas as pd
from statsmodels.tsa.stattools import adfuller , kpss
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import pmdarima as pm
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, mean_squared_error
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame as SparkDataFrame


# COMMAND ----------

#Lectura tabla silver y creacion dataframe
spark_df = spark.read.format("delta").table("Silver.AST_VentasProcesadas")
df_simplificado = spark_df.toPandas()

# Muestra las primeras filas del DataFrame
print(df_simplificado.head())

# COMMAND ----------

#df_simplificado['FechaVenta'] = pd.to_datetime(df_simplificado['FechaVenta'])
df_simplificado.set_index('FechaVenta', inplace=True)

# COMMAND ----------

df_simplificado.head()

# COMMAND ----------

df_simplificado.shape

# COMMAND ----------

# Crear una máscara para identificar columnas con al menos el 50% de ceros
#mask = (df_simplificado == 0).mean() >= 0.8

# COMMAND ----------

# Filtrar las columnas que no cumplen con la condición
#df_simplificado_sin_cero = df_simplificado#.loc[:, ~mask]

# COMMAND ----------

#df_simplificado_sin_cero

# COMMAND ----------


fecha_inicio_to = '2016-01-01'
fecha_fin_to = '2024-06-30'

# Filtrar los datos entre las fechas prediccion
df_simplificado_agrupado = df_simplificado.loc[fecha_inicio_to:fecha_fin_to]

# Cambiar el índice (que es la fecha) a una columna
df_simplificado_agrupado.reset_index(inplace=True)


# Renombrar la columna del índice de fecha (opcional)
df_simplificado_agrupado.rename(columns={'index': 'FechaVenta'}, inplace=True)

#Agrupar los datos por mes
df_simplificado_agrupado = df_simplificado_agrupado.resample('M', on='FechaVenta').sum().reset_index()

# COMMAND ----------

df_simplificado_agrupado

# COMMAND ----------

# Seteo del indice
df_simplificado_agrupado['FechaVenta'] = pd.to_datetime(df_simplificado_agrupado['FechaVenta'])
df_simplificado_agrupado.set_index('FechaVenta', inplace=True)

# COMMAND ----------

df_simplificado_agrupado

# COMMAND ----------


# Lista para almacenar los resultados
results = []

# Función para realizar la prueba ADF
def adf_test(series, column_name):
    # Verificar si la serie tiene valores constantes
    if series.max() == series.min():
        results.append({'Columna': column_name, 'Estacionaria': 'Error: La serie temporal es constante'})
        return
    try:
        result = adfuller(series)
        adf_statistic = result[0]
        p_value = result[1]
        critical_values = result[4]
        # Determinar si la serie es estacionaria
        if p_value < 0.05:
            estacionaria = 'Si'
        else:
            estacionaria = 'No'
        # Almacenar los resultados
        results.append({
            'Columna': column_name,
            'Estacionaria': estacionaria,
            'Estadístico ADF': adf_statistic,
            'Valor p': p_value,
            'Valores críticos': critical_values
        })
    except Exception as e:
        results.append({'Columna': column_name, 'Estacionaria': f'Error: {e}'})

# Iterar sobre las columnas del DataFrame
for col in df_simplificado_agrupado.columns:
    adf_test(df_simplificado_agrupado[col], col)

# Convertir la lista de resultados en un DataFrame
df_resultados = pd.DataFrame(results)

# Mostrar el DataFrame de resultados
print(df_resultados)

# COMMAND ----------

df_resultados.head()

# COMMAND ----------

# Filtrar las columnas no estacionarias
non_stationary_columns = df_resultados[df_resultados['Estacionaria'] == 'No']['Columna']

# Eliminar las columnas no estacionarias del DataFrame df_simplificado
df_simplificado_estacionarias = df_simplificado_agrupado.drop(columns=non_stationary_columns)

# Mostrar el DataFrame limpio
print(df_simplificado_estacionarias)

# COMMAND ----------

df_simplificado_estacionarias_tmp

# COMMAND ----------

# Cambiar el índice (que es la fecha) a una columna
df_simplificado_estacionarias_tmp= df_simplificado_estacionarias

# Crear una SparkSession si aún no existe
spark_gold_est = SparkSession.builder.getOrCreate()

# Convertir el DataFrame de pandas a un DataFrame de Spark
df_spark_silver_est = spark_gold_est.createDataFrame(df_simplificado_estacionarias_tmp)

df_spark_silver_est.write.format("delta").mode("overwrite").saveAsTable("Silver.AST_VentasProcesadasEstacionarias")

# COMMAND ----------

#Separa los dataframe de entrenamiento y test
fecha_corte = '2022-10-31'
df_train=df_simplificado_estacionarias[df_simplificado_estacionarias.index <= fecha_corte]
df_test=df_simplificado_estacionarias[df_simplificado_estacionarias.index> fecha_corte]

# COMMAND ----------

num_meses_predecir=20

# COMMAND ----------

df_train.head()

# COMMAND ----------

df_train.tail()

# COMMAND ----------

df_test.head()

# COMMAND ----------

df_test.tail()

# COMMAND ----------

# MAGIC %md
# MAGIC SARIMA

# COMMAND ----------


# Crear un diccionario para almacenar los modelos, predicciones y métricas
predictions_sarima_ajustado = {}

metricas_sarima = pd.DataFrame(columns=['Producto', 'SAR-MAE', 'SAR-RMSE', 'SAR-MAPE','SAR-R2'])

# Iterar sobre las columnas del DataFrame
for col in df_train.columns:

  # Ajustar el modelo usando auto_arima para encontrar los mejores parámetros
  model_auto_arima = pm.auto_arima(df_train[col], 
                                  seasonal=False, 
                                  m=num_meses_predecir,
                                  trace=True,
                                  stepwise=True,                                   
                                  suppress_warnings=True)

   # Ajustar el modelo ARIMA con los parámetros encontrados
  p, d, q = model_auto_arima.order
  P, D, Q, s = model_auto_arima.seasonal_order

  model_sarima  = SARIMAX(df_train[col], order=(p, d, q), seasonal_order=(P, D, Q, s))
  model_sarima_fit = model_sarima.fit()
  
 # Hacer predicciones para el número de meses 
  forecast = model_sarima_fit.forecast(steps=num_meses_predecir)

  # Almacenar las predicciones en el diccionario
  predictions_sarima_ajustado[col] = forecast

  # Calcular las métricas
  actual = df_test[col].values[:num_meses_predecir]  # Asegúrate de que tengas suficientes datos en df_test
  mae = mean_absolute_error(actual, forecast)
  rmse = np.sqrt(mean_squared_error(actual, forecast))
  mape = np.mean(np.abs((actual - forecast) / actual)) * 100
  r2 = 1 - (np.sum((actual - forecast) ** 2) / np.sum((actual - np.mean(actual)) ** 2))

  # Crear un DataFrame temporal con las métricas para este producto
  metricas_tempsa = pd.DataFrame({
        'Producto': [col],
        'SAR-MAE': [mae],
        'SAR-RMSE': [rmse],
        'SAR-MAPE': [mape],
        'SAR-R2': [r2]
   })

  # Usar pd.concat() para agregar las métricas al DataFrame general
  metricas_sarima = pd.concat([metricas_sarima, metricas_tempsa], ignore_index=True)



# COMMAND ----------

predictions_sarima_df = pd.DataFrame()

# Iterar sobre las columnas del diccionario de predicciones
for col, forecast in predictions_sarima_ajustado.items():
    # Crear un rango de fechas para las predicciones
    last_date = df_train.index[-1]  # La última fecha del DataFrame de entrenamiento
    forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=num_meses_predecir, freq='M')

    # Asignar el rango de fechas como índice y agregar las predicciones al DataFrame
    predictions_sarima_df[col] = forecast.values
    predictions_sarima_df.index = forecast_dates


# COMMAND ----------

# Cambiar el índice (que es la fecha) a una columna
predictions_sarima_df.reset_index(inplace=True)

# Renombrar la columna del índice de fecha (opcional)
predictions_sarima_df.rename(columns={'index': 'FechaVenta'}, inplace=True)

# COMMAND ----------

predictions_sarima_df

# COMMAND ----------

predictions_sarima_df.columns

# COMMAND ----------

# MAGIC %md
# MAGIC PROPHET

# COMMAND ----------

# Crear un diccionario para almacenar los modelos y las predicciones
predicciones_prophet = pd.DataFrame()

# Crear un DataFrame para almacenar las métricas de evaluación
metricas_prophet = pd.DataFrame(columns=['Producto', 'PRO-MAE', 'PRO-RMSE', 'PRO-MAPE','PRO-R2'])

# Ajustar el modelo Prophet para cada columna y hacer predicciones
for col in df_train.columns:
  # Crear un DataFrame compatible con Prophet
  df_prophet = df_train[[col]].reset_index()  
  df_prophet = df_prophet.rename(columns={'FechaVenta': 'ds', col: 'y'})
      
  # Ajustar el modelo Prophet
  model_prophet = Prophet()
  model_prophet.add_seasonality(
    name='monthly',
    period=30.5,
    fourier_order=5  # ajusta este valor según la complejidad de tus datos
  )
  model_prophet_fit= model_prophet.fit(df_prophet)  
  #Hacer predicciones para el numero de meses
  future = model_prophet_fit.make_future_dataframe(periods=(num_meses_predecir+1), freq='M')
  forecast = model_prophet_fit.predict(future)
  # Reemplazar valores negativos por cero
  forecast['yhat'] = forecast['yhat'].apply(lambda x: max(x, 0))
  
  # Almacenar las predicciones en el diccionario
  predicciones_prophet[f'{col}'] = forecast['yhat']    
  # Extraer las predicciones solo para el periodo de test
  pred_test = forecast['yhat'][-num_meses_predecir:]
    
  # Calcular las métricas de evaluación
  mae = mean_absolute_error(df_test[col], pred_test)
  rmse = np.sqrt(mean_squared_error(df_test[col], pred_test))
  mape = np.mean(np.abs((df_test[col] - pred_test) / df_test[col])) * 100  
  r2 = r2_score(df_test[col], pred_test)
    
  # Crear un DataFrame temporal con las métricas para este producto
  metricas_temp = pd.DataFrame({
        'Producto': [col],        'PRO-MAE': [mae],        'PRO-RMSE': [rmse],        'PRO-MAPE': [mape],        'PRO-R2': [r2]
   })

  # Usar pd.concat() para agregar las métricas al DataFrame general
  metricas_prophet = pd.concat([metricas_prophet, metricas_temp], ignore_index=True)


# COMMAND ----------

#Asigno los valores
predicciones_prophet_df = predicciones_prophet

# Añadir la columna de fechas al DataFrame
predicciones_prophet_df['FechaVenta'] = forecast['ds'] 

# Reorganizar las columnas para que 'FechaVenta' esté al principio
predicciones_prophet_df = predicciones_prophet_df[['FechaVenta'] + [col for col in predicciones_prophet_df.columns if col != 'FechaVenta']]


# COMMAND ----------

#Setear indice la FechaVenta
predicciones_prophet_df.set_index('FechaVenta', inplace=True)

# COMMAND ----------

predicciones_prophet_df

# COMMAND ----------

fecha_inicio_ph = '2022-11-01'
fecha_fin_ph = '2024-06-30'

# Filtrar los datos entre las fechas prediccion
predicciones_prophet_filtrado_df = predicciones_prophet_df.loc[fecha_inicio_ph:fecha_fin_ph]

# COMMAND ----------

# Cambiar el índice (que es la fecha) a una columna
predicciones_prophet_filtrado_df.reset_index(inplace=True)

# Renombrar la columna del índice de fecha (opcional)
predicciones_prophet_filtrado_df.rename(columns={'index': 'FechaVenta'}, inplace=True)

# COMMAND ----------

predicciones_prophet_filtrado_df

# COMMAND ----------

# Agrega una columna que indique el nombre del DataFrame
predictions_sarima_df['Origen'] = 'SARIMA'  
predicciones_prophet_filtrado_df['Origen'] = 'PROPHET'  

# COMMAND ----------

# Alinear las columnas para asegurarse de que coincidan
predictions_sarima_df = predictions_sarima_df[predicciones_prophet_filtrado_df.columns] 

# Unir los DataFrames
df_unido = pd.concat([predictions_sarima_df, predicciones_prophet_filtrado_df], ignore_index=True)


# COMMAND ----------

df_unido

# COMMAND ----------

# Crear una SparkSession si aún no existe
spark_gold = SparkSession.builder.getOrCreate()

# Convertir el DataFrame de pandas a un DataFrame de Spark
df_spark_silver = spark_gold.createDataFrame(df_unido)

df_spark_silver.write.format("delta").mode("overwrite").saveAsTable("Gold.AST_VentasPrediccionesArimaProphet")

# COMMAND ----------

# MAGIC %md
# MAGIC EVALUACION METRICAS

# COMMAND ----------

metricas_unificadas = pd.merge(metricas_sarima, metricas_prophet, on='Producto')

# COMMAND ----------

# Aumentar el límite de filas y columnas a mostrar
#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)

# Imprimir el DataFrame completo
productos_especificos = ['TL-WR840N', 'ARCHER_C50','TL-WR841HP','ARCHER_C64']

# Filtrar el DataFrame
metricas_espe = metricas_unificadas[metricas_unificadas['Producto'].isin(productos_especificos)]

metricas_espe.head()

# Restaurar la configuración original (opcional)
#pd.reset_option('display.max_rows')
#pd.reset_option('display.max_columns')

# COMMAND ----------

metricas_unificadas

# COMMAND ----------

# Crear una SparkSession si aún no existe
spark_gold_m = SparkSession.builder.getOrCreate()

# Convertir el DataFrame de pandas a un DataFrame de Spark
df_spark_silver_m = spark_gold_m.createDataFrame(metricas_unificadas)

df_spark_silver_m.write.format("delta").mode("overwrite").saveAsTable("Gold.AST_Metricas")