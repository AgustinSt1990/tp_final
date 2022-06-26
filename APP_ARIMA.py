# ACTUALIZAR
#print ('>> APP ARIMA: Recibe una serie, y aplica logaritmo, 3 procesos de diferenciacion y un proceso de optimización, luego grafica para visualizar resultados y seleccionar parámetros ARIMA')


from os import times
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


import statsmodels.api as sm
import statsmodels.tsa.api as smt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

import logging
import datetime as dt
formatter = '%(levelname)s:\n %(message)s | %(asctime)s\n| line %(lineno)d\n'
logging.basicConfig(filename='logfile_arima.log', level=logging.INFO, force=True, filemode='w', **{'format':formatter})
logger = logging.getLogger()

def get_stationarity(timeseries, windows=12, visualize=False):
    """Recibe un DataFrame univariado con index datetime, y devuelve el p-value del adfuller test
    Tiene la función de visualizar lo cual grafica e imprime la serie, la SMA, y la fluctuación (volatility)"""
    logger.info('{}) FUNCTION: get_stationarity'.format(x_1()))
    
    if visualize:
        if windows != None:
            # test: rolling statistics
            rolling_mean = np.exp(timeseries.ewm(windows).mean())
            rolling_std = np.exp(timeseries.ewm(windows).std())
            
            # rolling statistics plot
            plt.figure(figsize=(20,7))
            plt.plot(np.exp(timeseries), color='orange', label='Original')
            plt.plot(rolling_mean, color='red', label='Rolling Mean')
            plt.plot(rolling_std, color='black', label='Rolling Std')
            plt.legend(loc='best')
            plt.title('Rolling Mean & Standard Deviation - window = {}'.format(windows))
            logger.info('{}) En la función get_stationarity, la línea siguiente es plt.show() y tiene un parámetro "block" - True para visualizar en VSCode, False para cerrar graphs cuando termina ejecución.'.format(x_1()))
            plt.show(block=False)
    
    # test 2: Dickey–Fuller test:
    result = adfuller(timeseries)
    if visualize:
        print ('\tDICKEY-FULLER TEST')
        print('ADF Statistic: {}'.format(result[0]))
        print('p-value: {}'.format(result[1]))
        print('Critical Values:')
        for key, value in result[4].items():
            print('\t{}: {}'.format(key, value))
    return result[1]

def test_stationarity(timeseries, windows=12, visualize=True):
    logger.info('{}) FUNCTION: test_stationarity'.format(x_1()))
    
    if isinstance(timeseries, type(pd.DataFrame())):
        logger.warning('{}) El algoritmo asume siempre la primera columna en el adfuller test'.format(x_1()))
        timeseries = timeseries[timeseries.columns.values[0]]
    df_log = np.log(timeseries)

    # test: SMA, EMA, shifted log value
    df_log_1 = (df_log  - df_log.rolling(windows).mean()).dropna()
    df_log_2 = (df_log - df_log.ewm(windows).mean()).dropna()
    df_log_3 = (df_log - df_log.shift()).dropna()

    differentiation_methods = [df_log_1, df_log_2, df_log_3]
    differentiation_methods_names = ['SMA', 'EMA', 'shifted_log_value']
    lista_p_values = []
    for method in differentiation_methods:
        lista_p_values.append(adfuller(method)[1])

    # seleccionar el metodo que tenga menor p-value	
    selected_method_id = pd.Series(lista_p_values).idxmin()

    # optimizar el método con nuevos parámetros
    lista_movingAvg_values = [2, 5, 9, 12, 14, 15, 16, 17, 20, 50, 55]
    lista_p_values_i = []
    if differentiation_methods_names[selected_method_id] == 'SMA':
        for i in lista_movingAvg_values:
            df_log_1 = (df_log - df_log.rolling(i).mean()).dropna()
            lista_p_values_i.append(get_stationarity(df_log_1))
        optmized_params_id = pd.Series(lista_p_values_i).idxmin()
        result = (df_log - df_log.rolling(lista_movingAvg_values[optmized_params_id]).mean()).dropna()
    elif differentiation_methods_names[selected_method_id] == 'EMA':
        for i in lista_movingAvg_values:
            df_log_2 = (df_log - df_log.ewm(i).mean()).dropna()
            lista_p_values_i.append(get_stationarity(df_log_2))
        optmized_params_id = pd.Series(lista_p_values_i).idxmin()
        result = (df_log - df_log.ewm(lista_movingAvg_values[optmized_params_id]).mean()).dropna()
    else:
        lista_movingAvg_values = [None]
        optmized_params_id = 0
        result = df_log_3
    
    print ('\tDIFFERENTIATION  PROCESS')
    print (f'Selected Method: {differentiation_methods_names[selected_method_id]}'\
        f'\nOptimized Params: {lista_movingAvg_values[optmized_params_id]}\n')
    get_stationarity(result, windows=lista_movingAvg_values[optmized_params_id], visualize=visualize)
    return np.exp(result)


# GridSearch ARIMA: https://machinelearningmastery.com/grid-search-arima-hyperparameters-with-python/

# Manual setings of params ARIMA
def params_set_plot(y, lags=None, figsize=(12, 7), style='bmh', dpi=100):
    """ 
        Plotea la serie de tiempo, el ACF y PACF y el test de Dickey–Fuller
        
        y - serie de tiempo
        lags - cuántos lags incluir para el cálculo de la ACF y PACF
        
    """
    if not isinstance(y, pd.Series):
        try:
            y = pd.Series(y)
        except:
            y = y[y.columns.values[0]]
        
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize, dpi=dpi)
        layout = (2, 2)
        
        # definimos ejes
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        
        y.plot(color='orange', ax=ts_ax)
        
        # obtengo el p-value con h0: raiz unitaria presente
        p_value = sm.tsa.stattools.adfuller(y)[1]
        
        ts_ax.set_title('Análisis de la Serie de Tiempo\n Dickey-Fuller: p={0:.5f}'\
                        .format(p_value))
        
        # plot de autocorrelacion
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        # plot de autocorrelacion parcial
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()
        plt.show(block=True)


block_1 = True
def x_1():
    global block_1
    try:
        if block_1: 1/0
        file = open('enumerate_app_arima.log', 'r')
        i = file.read()
        file.close()
        i = int(i)
        i += 1
        i = str(i)
        file = open('enumerate_app_arima.log', 'w')
        file.write(i)
        file.close()  
        return i
    
    except:
        file = open('enumerate_app_arima.log', 'a')
        file.close()
        file = open('enumerate_app_arima.log', 'w')
        file.write('1')
        file.close()
        block_1 = False
        return '1'

if __name__ == '__main__':
    logger.info('{}) START'.format(x_1()))
    logger.warning('{}) No está funcionando la funcion load_dataset()'.format(x_1()))
    df = pd.read_csv('AirPassengers.csv')

    df['Month'] = pd.to_datetime(df['Month'])
    df['Passengers'] = df['#Passengers'].astype(float)
    df.drop(columns='#Passengers', inplace=True)
    df.set_index('Month', inplace=True)
    df = df.asfreq('MS')
    logger.critical('{}) sobre la función .asfreq(), puse "M" en vez de "MS" y me arrojó la serie completa de NaNs. Cuidado al definir frecuencias')

    

    # Se evalua la estacionalidad con 3 procesos diferentes
    logger.critical('{}) En este algoritmo aplicamos 3 procesos de diferenciacion sobre el logaritmo del dataframe respecto a 3 transformaciones, SMA, EMA, shift'.format(x_1()))
    logger.critical('{}) X!EL PROCESO DE DIFERENCIACION en la notebook ARIMA se llama "residuo" y es la diferencia entre el valor real y el predicho por el modelo reg_log_est'.format(x_1()))
    diferenciacion = test_stationarity(df)
    print (type(df))# type: dataframe de 1 columna

   

    #residuos = valor_real - predicciones_modelo_reg_log_est# no esta desarrollado
    # get_stationarity(residuos)

    params_set_plot(diferenciacion)

    # verificacion manual de los parametros ARIMA(p,d,q)




