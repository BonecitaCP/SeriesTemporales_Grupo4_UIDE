# Databricks notebook source
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame as SparkDataFrame

# COMMAND ----------

#Lectura tabla silver y creacion dataframe
spark_df = spark.read.format("delta").table("Silver.AST_VentasProcesadasEstacionarias")
df_estacionarias = spark_df.toPandas()

#Asignamos una nueva columna para identificar que son los valores originales
df_original= df_estacionarias
df_original['Origen'] = 'ORIGINAL'  


fecha_desde = pd.to_datetime('2016-01-01')
fecha_hasta = pd.to_datetime('2024-06-30')

# Filtrar el DataFrame por el rango de fechas
df_original = df_original[(df_original['FechaVenta'] >= fecha_desde) & (df_original['FechaVenta'] <= fecha_hasta)]

'''
# Cambiar el índice (que es la fecha) a una columna
df_original.reset_index(inplace=True)

# Renombrar la columna del índice de fecha (opcional)
df_original.rename(columns={'index': 'FechaVenta'}, inplace=True)'''

# COMMAND ----------

#Lectura tabla silver y creacion dataframe
spark_df = spark.read.format("delta").table("Gold.AST_VentasPrediccionesArimaProphet")
df_predicciones = spark_df.toPandas()

df_filtrado_predicciones = df_predicciones[df_predicciones['Origen'] == 'PROPHET']

# COMMAND ----------

# Alinear las columnas para asegurarse de que coincidan
df_total = df_filtrado_predicciones[df_original.columns] 

# Unir los DataFrames
df_total = pd.concat([df_filtrado_predicciones, df_original], ignore_index=True)

# COMMAND ----------

df_total.columns

# COMMAND ----------

product_columns_total = df_total.columns[1:75]

# COMMAND ----------

product_columns_total

# COMMAND ----------

df_pivot = pd.melt(df_total, 
                   id_vars=['FechaVenta', 'Origen'],  # Columnas que se mantendrán fijas
                   value_vars=product_columns_total,  # Columnas que quieres pivotear
                   var_name='Producto',  # Nuevo nombre de la columna para los productos
                   value_name='Cantidad')  # Nuevo nombre de la columna de cantidades

# COMMAND ----------

df_pivot.rename(columns={'FechaVenta': 'Fecha'}, inplace=True)

# COMMAND ----------

df_pivot

# COMMAND ----------

# Crear una SparkSession si aún no existe
spark_total= SparkSession.builder.getOrCreate()

# Convertir el DataFrame de pandas a un DataFrame de Spark
df_spark_total = spark_total.createDataFrame(df_pivot)

df_spark_total.write.format("delta").mode("overwrite").saveAsTable("Gold.AST_VentasPrediccionesOriginalProphetPivot")

# COMMAND ----------

pip  install openpyxl

# COMMAND ----------

df_pivot.to_excel("/Workspace/Prueba/OriginalesyResultadosUnificados.xlsx", index=False)
