# <font color = #001d8c><u>PRESENTACION FINAL"MODELOS DE PRONÓSTICO TEMPORAL"</u></font>

Agustín Federico Stigliano y compañia, entregado para Digital House.

## <font color = #0024af>1. <u>Documentos Frontend</u></font>

<b>Presentación Técnica:</b>

- <b>TP_FINAL_Data_Science.ipynb</b>: Es donde las bases de datos iniciales son tratadas y estudiadas para luego extraer muestras que utilizaremos para machine learning.

- <b>TP_FINAL_Machine_Learning.ipynb</b>: Recibe la selección de muestras, procesa algoritmos de machine learning y exporta los modelos entrenados optimizados

<b>Presentación Audiencia:</b>

- <b>Presentacion_TP_FINAL_Grupo_1.pdf</b>: Filminas realizadas para la presentación al CEO de la empresa.
    
## <font color = #0024af>2. <u>Documentos Backend</u></font>

- Aplicaciones:
    - <b>APP_ARIMA.py</b>: Estudio de test estacionalidad habiendo procesado 3 diferenciaciones estandar previamente, y contiene funciones para visualizar los valores de parámetros para modelo ARIMA.
    - <b>APP_REG_LOG_EST.py</b>: son 6 modelos de regresión lineal, algunos con transformación logaritmica y otros con variables estacionarias.
    - <b>APP_XGBOOST.py</b>: es un pseudo-modelo de aprendizaje supervisado porque predice de forma naive, no queda entrenado. La serie de tiempo se transforma a un problema predictivo, cosa que luego se itera reiteradas veces. 
    
- Librerias:
    - <b>SmartPandas.py</b>: La nueva versión contiene funciones que se utilizan para generar instancias numeradas, que luego se añaden dentro de los mensaje de la libreria logging.
    - <b>SmartSeries.py</b>: Contiene funciones para cargar dataset (análogo a pd.read_csv) especial para series de tiempo, contiene clases para el tratamiento de los datasets.
    
- Datos:
    - customer_transaction_data.csv: se utiliza en la notebook de Data Science
    - item_data.csv: se utiliza en la notebook de Data Science
    
- Auxiliares:
    - requierements.txt: son las librerias utilizadas y las versiones de diseño

<br>

<b>A modo de Spoiler les dejo unos gráficos:</b>

Los datos llegan hasta medidados del 2013, y en verde se puede ver la predicción de los modelos para el segundo semestre del 2013. Se hicieron algunas aclaraciónes de forma manual.

![MEAT](https://user-images.githubusercontent.com/95892143/177053014-aef61feb-d9a2-4112-a5b7-166261099ed6.png)

![farma](https://user-images.githubusercontent.com/95892143/177053040-518206d4-2e18-4463-b39c-46bbd5873511.png)


