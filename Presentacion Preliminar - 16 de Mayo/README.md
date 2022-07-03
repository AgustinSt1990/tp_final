# <font color = #001d8c><u>PRESENTACION DEL PROYECTO "TRABAJO PRACTICO FINAL"</u></font>

**Digital House**

**Grupo 1**
- Integrantes del grupo:
    - Gonzalo Barbot - Project Manager
    - Agustin Stigliano - Code Development Manager
    - Fernando Dupont - Presentation Development Manager
    
Acceso al repositorio: https://github.com/AgustinSt1990/tp_final    

## <font color = #0024af>1. <u>Tema de investigación</u></font>

Se trata de datos relevados de una empresa con información relacionada al área comercial.

Partimos de un conjunto de datasets provenientes de kaggle que se encuentan en el link a continuacion:
    
`https://www.kaggle.com/datasets/vasudeva009/coupon-redemption-smote-feature-selection`

- Los datasets en cuestion son:
    - customer_transaction_data.csv
    - item_data.csv
    - customer_demographics.csv
    - coupon_item_mapping.csv
    - campaing_data.csv
    
A continuación se muestra un gráfico en el que se conectan los dataset recién mencionados:

_Aclaración: El enfoque que se le pretende dar al trabajo consiste en transformar los datos comerciales de una empresa a través de algoritmos de machine learning, robustos, en información valiosa para los ejecutivos de la empresa. Que incluso tenga el potencial de convertirse en una herramienta que permita desarrollar, en parte, las presentaciones pertinentes para la toma de decisiones, desde el ejecutivo al órgano directivo de la empresa._

<img width="848" alt="Schema_datasets" src="https://user-images.githubusercontent.com/95892143/173867516-52a16585-9097-47fa-882e-f1c20626c49f.png">

### <font color = #f44611><u>Presentación de la propuesta</u></font>

#### Puesta en común

La atención la se centra en el dataset "Customer Transaction Data" : "transaction_data".

#### <u>Estudio de columnas</u>: transaction_data

- 0	date: Fecha en que ocurrió la transacción
- 1	customer_id: conexión con dataset "customer_data"
- 2	quantity: Cantidad de "item_id" adquiridos en la transacción
- 3	item_id: conexión con dataset "item_data"
- 4	coupon_discount: descuento aplicado a la transacción
- 5	other_discount: descuento aplicado a la transacción
- 6	selling_price: precio de venta unitario del "item"


Hay 2 datasets que consideramos pertinentes para ampliar el rango de aprendizaje de los datos. {'customer_demographics_data': 'customer_data', 'item_data': 'item_data'}

#### <u>Estudio de columnas</u>: customer_data
- 0	rented: 0 establecimiento no alquilado - 1 establecimiento alquilado
- 1	income_bracket: Categórica ordinal - los labels mayores corresponden a ingresos mayores
- 2	customer_id: conexión con "transaction_data"
- 3	marital_status: contienen NaNs, irrelevante
- 4	no_of_children: contiene NaNs, irrelevante
- 5	family_size: cantidad de familiares
- 6	age_range: Categórica, los labels informar el rango etario de la familia

#### <u>Estudio de columnas</u>: item_data

Podemos ver que encontramos dos variables categóricas, una es binaria y la otra multiclase, una corresponde a "brand_type" y la otra a "category"

- brand_type: se divide entre una marca local o una establecida 
- category: se dividen en 19 categorias de locales a los cuales la empresa vende sus productos. Dichas categorias responden a categoría de "cliente"
- brand: 5528 marcas
- item_id: conexion con transaction_data

Luego se genera el dataset "data" con el que pretendemos utilizar de punto de partida para generar otras perspectivas mediante el uso de tablas, agrupamientos, rankings, etc. 

Entonces pasamos de tener esto:

![data_raw](https://user-images.githubusercontent.com/95892143/173867627-b960f41e-6677-42b1-b15e-60048c55602a.png)

a tener esto:

![data_merged](https://user-images.githubusercontent.com/95892143/173867667-c34673e8-6dee-4a91-b4e8-d4daba06895c.png)


#### Formalización del proyecto

La finalidad última del trabajo, es entrenar algoritmos de machine learning con series de tiempo. Queremos predecir "ventas futuras", pero también queremos identificar variaciones en la varianza de los indicadores. 

Apuntamos a entrenar modelos de machine learning que nos permitan predecir la demanda de cada categoría y la probabilidad de qué items serán demandados, apuntando siempre a un plazo de 2 meses hacia adelante, así poder evaluar el curso de acción y tomar decisiones con una ventana de tiempo razonable, y esperamos desarrollar un algorítmo que ayude en eso. A recibir la información en el momento que la información vale.

## <font color = #0024af>2. <u>Antecedentes sobre el tema</u></font>

El comercio es un campo inherente a la vida cotidiana. Ya sea desde el lado activo o desde el pasivo todos interactuamos con el comercio. La idea de entrenar algoritmos que predigan ventas puede ser adaptados a futuro a otras aplicaciones, como por ejemplo predecir compras. O mejor aún, pueden ser transformados para convertirse en sofisticadas herramientas de estudio y seguimiento del precio de activos financieros, y convertirse fuentes de ingreso gracias y a través del comercio de dicho activo.

Para encarar ese desafío proyectamos hacer uso de librerias de Python con statmodels para aplicar las 2 clases de tratamiento para las series de tiempo:

- Time series analysis: análisis de series de tiempo

- Time series forecasting: Pronóstico de series de tiempo


## <font color = #0024af>3. <u>Aporte esperado</u></font>

El aporte que esperamos lograr es que nuestros modelos sean eficientes, que corran a una velocidad acorde al rango temporal de la predicción (en este caso a 2 meses no habria problema). Esperamos desarrollar algoritmos automatizados, escalables y que cumplan con los criterios DRY y todas las buenas prácticas de programación. Queremos aprovechar esta instancia para crear clases con POO que nos permitan generar ensambles robustos, y reutilizables, y aprovechar al máximo el potencial de pipelines en la redacción del código.

También tenemos presente el hecho didáctico del trabajo y queremos que la presentación final sea fácilmente legible sus resultados, por fuera de la eficiencia, proponemos que el código genere contenido pertinente para la elaboración futura de ensayos realizados con el código y que permita presentar las métricas y los resultados de forma simplificada.

## <font color = #0024af>4. <u>Disponibilidad de datos e infraestructura</u></font>

Los dataset están todos en el link publicado en la introducción de este texto.

Las librerias son todas aquellas con las que venimos trabajando, Pandas, Numpy, Scipy, Pickle, Scikit Learn, statsmodels, xgboost, matplotlib, seaborn.

Otras librerías que consideramos usar es Arrow, scikit-image, pyspark, mlflow, PyGraphviz, LightGBM, Bokeh, logging, os, etc.

## <font color = #0024af>5. <u>Plan de trabajo y cronograma tentativo</u></font>

Semana 1: Limpieza, y selección de features. Categorización.

Semana 2: Entrenamiento de modelos para evaluar features. Primera pruebas de modelos predictivos sobre "target"

Semana 3: Estudio del feedback de los modelos. Recalibración de Features. Ajuste de hiperparámetros de modelos I.

Semana 4: Creación de pipelines. Ensambles. Obtención de primeras métricas a presentar.

Semana 5: Backtesting de los modelos. Limpieza, depuración y debbugin del código. Generar métricas para la presentación.

Semana 6: Encarar presentación. Desarrollar funciones para genera contenido visual en la presentación. Backtesting del código. Presentar.

Semana 7: Desarrollar la presentación. Dividir contenido. Realizar exposición.
