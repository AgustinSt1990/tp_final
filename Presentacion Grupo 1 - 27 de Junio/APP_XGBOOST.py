#print ('>> APP XGBOST: APLICACION XGBOOST PARA TIME SERIES FORECASTING\n')
import numpy as np
import pandas as pd

from tqdm import tqdm

import xgboost
from sklearn.metrics import mean_squared_error
print ('xgboost version actual:', xgboost.__version__)
print ('Version de diseño en 14/6/2022\n\txgboost version: 1.6.0')

import pickle

import logging
import datetime as dt
formatter = '%(levelname)s:\n %(message)s | %(asctime)s\n| line %(lineno)d\n'
logging.basicConfig(filename='logfile_xgboost.log', level=logging.INFO, force=True, filemode='w', **{'format':formatter})
logger = logging.getLogger()


def load_dataset(file_name):
    logger.info(f'LOADING FUNCTION: load_dataset(): se carga el dataset llamado {file_name}')
    df = pd.read_csv(file_name)
    df['date'] = pd.to_datetime(df['date'])
    logger.warning('El dataset debe tener una columna "date" con formato "yyyy-mm-dd"')
    df.set_index('date', inplace=True)
    df = df.asfreq('D')
    logger.warning('Por default se considera la serie con freq = "D"')
    print ('Cantidad de NaNs en ', df.isna().sum().to_string())
    if df.isna().sum().sum() > 0: df.dropna(inplace=True); print ('NaNs eliminados\n')
    return df

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    '''Transforma una serie de datos en una matriz de datos supervisados'''
    logger.info('SET DATA FUNCTION: series_to_supervised(): se transforma una serie de datos en una matriz de datos supervisados')
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols = list()
    # secuencia de entrada (t-n, ..., t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # secuencia de salida (t, t+1, ..., t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # concatenar todas las columnas
    agg = pd.concat(cols, axis=1)
    # quitar las filas que contienen NaN
    if dropnan: agg.dropna(inplace=True)
    return pd.DataFrame(agg.values, columns=['train', 'test'])


def walk_forward_validation(data, n_test, visualization=False):
    """Validacion walk-forward para datos univariados"""
    logger.info('MODEL FUNCTION: walk_forward_validation(): funcion principal - queda almacenada en "app_xgboost_forecast.pickle"')
    data = series_to_supervised(data.values, n_in=1, n_out=1, dropnan=True)
    predictions, progress = list(), list()
    # dividir dataset
    train, test = train_test_split_st(data, n_test)
    # generar un historial con los datos de entrenamiento
    history = train.copy()
    # iterar un paso por vez dentro del set de testeo
    for i in tqdm(range(len(test))):
        # dividir la fila de testeo en entrada y salida
        testX, testy = test.iloc[i, :-1], test.iloc[i, -1]
        # entrenar el modelo sobre el historial y realizar una prediccion
        yhat = xgboost_forecast(history, testX)
        # almacenar la prediccion en la lista de predicciones
        predictions.append(yhat)
        # agregar la observacion actual al historial para la siguiente iteracion
        history = pd.concat([history, test.iloc[i,:].to_frame().T], axis=0)
        # computar el progreso
        progress.append('>expected=%.1f, predicted=%.1f' % (testy, yhat))
    
    if visualization: print('\n'.join([i for i in progress]))
    # estimar el error por RMSE
    error = mean_squared_error(test.iloc[:, -1], predictions)
    df_test = test.iloc[:, 1].to_frame()
    df_test.columns = ['real']
    return error, df_test, pd.DataFrame(pd.Series(predictions), columns=['predictions'])


def train_test_split_st(data, n_test):
    """Divide un dataset univariado en train y test"""
    logger.info('SET DATA FUNCTION: train_test_split_st(): se divide un dataset univariado en train y test')
    return data.iloc[:-n_test, :], data.iloc[-n_test:, :]

def xgboost_forecast(history, testX):
    """Entrena un modelo xgboost y hace una prediccion "one step forward" """
    logger.info('MODEL FUNCTION: xgboost_forecast(): se entrena el modelo "xgboost N {}" y se realiza la prediccion de un paso'.format(x_1()))
    # transforma lista en array
    history = np.array(history)
    # divide en columnas de entrada y salida
    trainX, trainy = history[:, :-1], history[:, -1]
    # entrenar modelo
    model = xgboost.XGBRegressor(objective='reg:squarederror', n_estimators=1000)
    model.fit(trainX, trainy)
    # predecir "one step forward"
    yhat = model.predict([testX])
    return yhat[0]

def guardar_funcion():
    """Guarda la funcion en un archivo pickle"""
    logger.info('SAVE FUNCTION: guardar_funcion(): se guarda la funcion en un archivo pickle')
    with open('app_xgboost_forecast.pickle', 'wb') as f:
        pickle.dump(walk_forward_validation, f)

block_1 = True
def x_1():
    global block_1
    try:
        if block_1: 1/0
        file = open('enumerate_app_xgboost.log', 'r')
        i = file.read()
        file.close()
        i = int(i)
        i += 1
        i = str(i)
        file = open('enumerate_app_xgboost.log', 'w')
        file.write(i)
        file.close()  
        return i
    
    except:
        file = open('enumerate_app_xgboost.log', 'a')
        file.close()
        file = open('enumerate_app_xgboost.log', 'w')
        file.write('1')
        file.close()
        block_1 = False
        return '1'

if __name__ == '__main__':
    logger.info('STARTING PROGRAM') 
    print ('='*75+'\n\t\tPROGRAMA PRINCIPAL\n'+'='*75+'\n')
    
    
    df = load_dataset('target_meat.csv')
    print (df)
    print (df.index)
    error, df_test, df_pred = walk_forward_validation(df, n_test=45)
    #print ('RMSE: ', mean_squared_error(df_test, np.exp(df_pred)))
    print ('RMSE: ', error)
    guardar_funcion()
    print ('Finalizacion de la app - Gracias por haberla utilizado')