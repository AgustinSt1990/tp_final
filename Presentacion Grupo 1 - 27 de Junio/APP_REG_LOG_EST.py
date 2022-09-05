# print ('>> APP REG_LOG_EST: APLICACION PARA ENTRENAR Y EXPORTAR UN MODELO DE REGRESION LINEAL CON TRANSFORMACION LOGARITMICA APLICANDO ESTACIONALIDAD')
from sys import prefix
from turtle import down
from SmartSeries import *
import SmartSeries
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import logging
import datetime as dt

formatter = "%(levelname)s:\n %(message)s | %(asctime)s\n| line %(lineno)d\n"
logging.basicConfig(
    filename="logfile_reg_log_est.log",
    level=logging.INFO,
    force=True,
    filemode="w",
    **{"format": formatter}
)
logger = logging.getLogger()


def get_formula(target, lista_features):
    string_features = " + ".join(lista_features)
    return target + " ~ " + string_features


def est_log_trend(df_train, vector, umbral=None, EMA=False, EMA_arg=None):
    if umbral != None:
        mask = pd.Series(vector.targets).isin([umbral]).values
        vector.change_target(vector, umbral)

    model_target = "log_" + vector.target
    features = df_train.columns.drop([vector.target, model_target, "timeIndex_sq"])
    if EMA:
        features += EMA_arg
    argumento = get_formula(model_target, features)
    return smf.ols(formula=argumento, data=df_train)


def est_log_trend_sq(df_train, vector, umbral=None):
    if umbral != None:
        mask = pd.Series(vector.targets).isin([umbral]).values
        vector.change_target(vector, umbral)
    model_target = "log_" + vector.target
    features = df_train.columns.drop(
        [vector.target, model_target]
    )  # aca eliminast timeIndex
    argumento = get_formula(model_target, features)
    return smf.ols(formula=argumento, data=df_train)


def log_trend(df_train, vector, umbral=None):
    if umbral != None:
        mask = pd.Series(vector.targets).isin([umbral]).values
        vector.change_target(vector, umbral)
    model_target = "log_" + vector.target
    features = ["timeIndex"]
    argumento = get_formula(model_target, features)
    return smf.ols(formula=argumento, data=df_train)


def log_trend_sq(df_train, vector, umbral=None):
    if umbral != None:
        mask = pd.Series(vector.targets).isin([umbral]).values
        vector.change_target(vector, umbral)
    model_target = "log_" + vector.target
    features = ["timeIndex", "timeIndex_sq"]
    argumento = get_formula(model_target, features)
    return smf.ols(formula=argumento, data=df_train)


def lin_trend(df_train, vector, umbral=None):
    if umbral != None:
        mask = pd.Series(vector.targets).isin([umbral]).values
        vector.change_target(vector, umbral)
    model_target = vector.target
    features = ["timeIndex"]
    argumento = get_formula(model_target, features)
    return smf.ols(formula=argumento, data=df_train)


def lin_trend_sq(df_train, vector, umbral=None):
    if umbral != None:
        mask = pd.Series(vector.targets).isin([umbral]).values
        vector.change_target(vector, umbral)
    model_target = vector.target
    features = ["timeIndex_sq"]
    argumento = get_formula(model_target, features)
    return smf.ols(formula=argumento, data=df_train)


def available_models():
    return [
        est_log_trend,
        est_log_trend_sq,
        log_trend,
        log_trend_sq,
        lin_trend,
        lin_trend_sq,
    ]


def def_features(vector):  # obsoleto
    features_est_log_trend = vector.vector.columns.drop(
        ["log_" + vector.target, vector.target, "timeIndex_sq"]
    )
    features_est_log_trend_sq = vector.vector.columns.drop(
        ["log_" + vector.target, vector.target, "timeIndex_sq"]
    )
    features_log_trend = ["timeIndex"]
    features_log_trend_sq = ["timeIndex_sq"]
    features_lin_trend = ["timeIndex"]
    features_lin_trend_sq = ["timeIndex_sq"]
    return [
        features_est_log_trend,
        features_est_log_trend_sq,
        features_log_trend,
        features_log_trend_sq,
        features_lin_trend,
        features_lin_trend_sq,
    ]


def def_models():
    return [
        "log_trend_est",
        "log_trend_est_sq",
        "log_trend",
        "log_trend_sq",
        "lin_trend",
        "lin_trend_sq",
    ]


if __name__ == "__main__":

    target_meat = SmartSeries.load_dataset("target_meat.csv")

    vector_1 = SmartSeries.VectorBuilder(target_meat)

    vector_1.create()
    vector_1.treatment()

    # creación de dataframe con los datos de tratamiento y variables dummy

    df = pd.concat(
        [
            vector_1.vector,
            pd.get_dummies(
                pd.Series(
                    [i.strftime("%U") for i in vector_1.vector.index],
                    index=vector_1.vector.index,
                ),
                prefix="sem",
            ),
        ],
        axis=1,
    )

    df_train, df_test = train_test_split(df, shuffle=False, test_size=60)

    model = est_log_trend(df_train, vector_1)

    # model predict
    y_pred = model.predict(df_test)

    # model RMSE
    rmse = mean_squared_error(y_pred, df_test["log_" + vector_1.target])

    rmse_2 = mean_squared_error(np.exp(y_pred), df_test[vector_1.target])

    print(rmse_2)
