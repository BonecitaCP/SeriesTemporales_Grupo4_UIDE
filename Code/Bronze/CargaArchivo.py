# Databricks notebook source
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame as SparkDataFrame


# COMMAND ----------

pip  install openpyxl

# COMMAND ----------




# Ruta del archivo en el punto de montaje
file_path = "/dbfs/mnt/DataLake/DataFuente/routers.xlsx"


# Lee el archivo Excel con pandas
df = pd.read_excel(file_path)

# Muestra las primeras filas del DataFrame
print(df.head())

# COMMAND ----------

print(df.columns)

# COMMAND ----------

# Limpiar nombres de columnas
df.columns = [col.replace(' ', '_').replace(',', '_').replace(';', '_').replace('{', '').replace('}', '').replace('(', '').replace(')', '').replace('\n', '').replace('\t', '').replace('=', '_') for col in df.columns]


# COMMAND ----------

# Crear una SparkSession si aún no existe
spark = SparkSession.builder.getOrCreate()

# Convertir el DataFrame de pandas a un DataFrame de Spark
df_spark = spark.createDataFrame(df)

# COMMAND ----------

df

# COMMAND ----------

# MAGIC %sql 
# MAGIC drop table if exists Bronze.AST_Ventas

# COMMAND ----------

df_spark.write.format("delta").mode("overwrite").saveAsTable("Bronze.AST_Ventas")

# COMMAND ----------

# MAGIC %sql 
# MAGIC
# MAGIC --drop table Bronze.ast_ventas

# COMMAND ----------

# MAGIC %sql
# MAGIC /*
# MAGIC CREATE TABLE spark_catalog.Bronze.ast_ventas (
# MAGIC   `AÑO` BIGINT,
# MAGIC   MES BIGINT,
# MAGIC   CAPITULO BIGINT,
# MAGIC   SUBCAPITULO BIGINT,
# MAGIC   RUC STRING,
# MAGIC   RAZON_SOCIAL STRING,
# MAGIC   RAZON_SOCIAL_DIRECCION STRING,
# MAGIC   REFRENDO STRING,
# MAGIC   NUME_SERIE BIGINT,
# MAGIC   TIPO_AFORO STRING,
# MAGIC   COD_REGIMEN BIGINT,
# MAGIC   REGIMEN STRING,
# MAGIC   DISTRITO STRING,
# MAGIC   AGENTE_AFIANZADO STRING,
# MAGIC   REMITENTE STRING,
# MAGIC   NOTIFY STRING,
# MAGIC   EMBARCADOR_CONSIGNEE STRING,
# MAGIC   EMBARCADOR_CONSIGNEE_ADDRESS STRING,
# MAGIC   AGENCIA STRING,
# MAGIC   LINEA STRING,
# MAGIC   MANIFIESTO STRING,
# MAGIC   FECH_EMBAR TIMESTAMP,
# MAGIC   FECH_LLEGA TIMESTAMP,
# MAGIC   FECH_INGRESO TIMESTAMP,
# MAGIC   FECH_PAGO TIMESTAMP,
# MAGIC   FECH_SALIDA TIMESTAMP,
# MAGIC   FACTURA STRING,
# MAGIC   BL STRING,
# MAGIC   NAVE STRING,
# MAGIC   viaje STRING,
# MAGIC   almacen STRING,
# MAGIC   DEP_COMERCIAL STRING,
# MAGIC   PARTIDA BIGINT,
# MAGIC   TNAN BIGINT,
# MAGIC   TASA_ADVALOREM DOUBLE,
# MAGIC   DESC_ARAN STRING,
# MAGIC   DESC_COMER STRING,
# MAGIC   MARCAS STRING,
# MAGIC   CIUDAD_EMBARQUE STRING,
# MAGIC   PAIS_EMBARQUE STRING,
# MAGIC   PAIS_ORIGEN STRING,
# MAGIC   TIPO_CARGA DOUBLE,
# MAGIC   UNIDADES DOUBLE,
# MAGIC   TIPO_UNIDAD STRING,
# MAGIC   ESTADO_MERCANCIA STRING,
# MAGIC   KILOS_NETO DOUBLE,
# MAGIC   FOB DOUBLE,
# MAGIC   FLETE DOUBLE,
# MAGIC   SEGURO DOUBLE,
# MAGIC   CIF DOUBLE,
# MAGIC   CODIGO_LIBERACION BIGINT,
# MAGIC   COD_LIBERACION STRING,
# MAGIC   MONEDA DOUBLE,
# MAGIC   ADV_PAG_PARTIDA DOUBLE,
# MAGIC   ADV_LIQ_PARTIDA BIGINT,
# MAGIC   DER_ESP_PARTIDA DOUBLE,
# MAGIC   CARACTERISTICA STRING,
# MAGIC   PRODUCTO STRING,
# MAGIC   MARCA_COMERCIAL STRING,
# MAGIC   ANiO_PRODUCIDO DOUBLE,
# MAGIC   MODELO_MERCADERIA STRING,
# MAGIC   FOB_UNITARIO DOUBLE,
# MAGIC   MANIFIESTO_ADUANA STRING,
# MAGIC   VIA STRING,
# MAGIC   REGIMEN_TIPO STRING,
# MAGIC   INCOTERM STRING,
# MAGIC   consolidadora STRING,
# MAGIC   cod_provincia STRING,
# MAGIC   PROVINCIA STRING,
# MAGIC   formulario DOUBLE,
# MAGIC   form_via_envio STRING,
# MAGIC   flete2 DOUBLE,
# MAGIC   CFR DOUBLE,
# MAGIC   CIF2 DOUBLE,
# MAGIC   dias_salida BIGINT,
# MAGIC   `ESTADO_DECLARACIÓN` STRING,
# MAGIC   tipo_regimen STRING,
# MAGIC   partida_descripcion DOUBLE,
# MAGIC   capitulo_descripcion STRING)
# MAGIC USING delta
# MAGIC LOCATION '/mnt/DataLakeNuevo/DataDelta/Bronze/AST_Ventas'*/

# COMMAND ----------

# MAGIC %sql 
# MAGIC
# MAGIC select * from Bronze.AST_Ventas