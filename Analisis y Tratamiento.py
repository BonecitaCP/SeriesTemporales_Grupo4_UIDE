# Databricks notebook source
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame as SparkDataFrame

# COMMAND ----------

#Lectura de la tabla delta hacia dataframe pandas
spark_df = spark.read.format("delta").table("Bronze.AST_Ventas")
df = spark_df.toPandas()

# COMMAND ----------

df.shape

# COMMAND ----------

df.head()

# COMMAND ----------

df.tail()

# COMMAND ----------

df.info()

# COMMAND ----------

df.describe()

# COMMAND ----------

#Duplicados
duplicate_count = df.duplicated().sum()
print(f"Se encontraron {duplicate_count} filas duplicadas.")

# COMMAND ----------

df_duplicados=df.drop_duplicates(keep=False)
duplicate_count = df_duplicados.duplicated().sum()
print(f"Se encontraron {duplicate_count} filas duplicadas.")

# COMMAND ----------

#Filtrar información de remitente TPLINK
remitentes_unicos = df_duplicados['REMITENTE'].unique()
remitentes_unicos= np.sort(remitentes_unicos.astype(str))
palabras_clave_remitente= ['TP LINK','TP-LINK','TPLINK']
expresion_regular_remitente = '|'.join(palabras_clave_remitente)
df_filtro_router = df[df['REMITENTE'].str.contains(expresion_regular_remitente, case=False, na=False)]

# COMMAND ----------

# Filtrar las marcas vendidas por TPLINK
marcas_unicas = df_filtro_router['MARCA_COMERCIAL'].unique()
marcas_unicas
marcas_correctas= ['TP LINK', 'TP-LINK', 'TPLINK', 'T P LINK','TP-Link']
expresion_regular_marcas_correctas = '|'.join(marcas_correctas)
df_filtro_router['MARCA_COMERCIAL'] = df_filtro_router['MARCA_COMERCIAL'].fillna('')
df_filtro_router_marca = df_filtro_router[df_filtro_router['MARCA_COMERCIAL'].str.contains(expresion_regular_marcas_correctas, regex=True)]
df_filtro_router_marca['MARCA_COMERCIAL'].unique()

# COMMAND ----------

df_filtro_columnas= df_filtro_router_marca[['RAZON_SOCIAL','FECH_INGRESO', 'UNIDADES','CIF','CARACTERISTICA','PRODUCTO', 'MODELO_MERCADERIA']]
df_filtro_columnas.info()

# COMMAND ----------

#Crear diccionario, para estandarizar los nombres modelos y eliminar modelos que no son routers
spark_df_dicc = spark.read.format("delta").table("Bronze.AST_Diccionario")
df_dicc = spark_df_dicc.toPandas()

# COMMAND ----------


#Cruce diccionario y dataframe de datos
df_filtro_columnas['MODELO_MERCADERIA']=df_filtro_columnas['MODELO_MERCADERIA'].str.upper()
df_dicc['MODELO_ANT']=df_dicc['MODELO_ANT'].str.upper()
df_dicc['MODELO_NUE']=df_dicc['MODELO_NUE'].str.upper()
df_combined = df_filtro_columnas.merge(df_dicc, left_on='MODELO_MERCADERIA', right_on='MODELO_ANT', how='left')

# COMMAND ----------

#Eliminar registros incorrectos (switches, tarjetas)
df_combined_el = df_combined.loc[df_combined['BORRAR'] != 'SI']

# COMMAND ----------

#Verifica que todos los registros cruzaron OK con diccionario
df_combined_el.loc[df_combined_el['MODELO_NUE'].isna()]

# COMMAND ----------

#Verificar que se eliminaron registros incorrectos
registros_si = df_combined_el[df_combined_el['BORRAR'] == 'SI']
registros_si

# COMMAND ----------

# Se genera dataframe más pequeño solo columnas necesarias para series
df_series=df_combined_el[['FECH_INGRESO','MODELO_NUE','UNIDADES']]

# Se cambia nombre columnas
df_series.columns = ['FechaVenta','Producto','Unidades']

df_series.head()

# COMMAND ----------

# Trasponer el DataFrame al formato para series
df_series_traspuesto = df_series.pivot_table(index='FechaVenta', columns='Producto', values='Unidades', aggfunc='sum')
# Reiniciar el índice para que 'Fecha' vuelva a ser una columna
df_series_traspuesto.reset_index(inplace=True)
df_series_traspuesto.head()

# COMMAND ----------

#Reemplazo de Vacios por 0
df_series_traspuesto.fillna(0, inplace=True)
df_series_traspuesto.head()

# COMMAND ----------

df_series_traspuesto.columns = [col.replace(' ', '_').replace(',', '_').replace(';', '_').replace('{', '').replace('}', '').replace('(', '').replace(')', '').replace('\n', '').replace('\t', '').replace('=', '_') for col in df_series_traspuesto.columns]

# COMMAND ----------

# Crear una SparkSession si aún no existe
spark_silver = SparkSession.builder.getOrCreate()

# Convertir el DataFrame de pandas a un DataFrame de Spark
df_spark_silver = spark_silver.createDataFrame(df_series_traspuesto)

df_spark_silver.write.format("delta").mode("overwrite").saveAsTable("Silver.AST_VentasProcesadas")

# COMMAND ----------

# MAGIC %sql 
# MAGIC
# MAGIC select * from Silver.AST_VentasProcesadas