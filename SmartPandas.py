"""Este es el archivo que contiene funciones y clases"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.pipeline import FeatureUnion, _fit_transform_one, _transform_one
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MaxAbsScaler

"""Funcion que devuelve informacion de un DataFrame en formato tipo DataFrame"""

def data_info(data, name='data'):
    df = pd.DataFrame(pd.Series(data.columns))
    df.columns = ['columna']
    df.columns.name = f'info de {name}'
    #df.index.name = 'index'
    df['Nan'] = data.isna().sum().values
    df['pct_nan'] = round(df['Nan']/data.shape[0]*100,2)
    df['dtype']  = data.dtypes.values
    df['count'] = data.count().values
    df['pct_reg'] = ((data.notna().sum().values / data.shape[0])*100).round(2)
    df['count_unique'] = [len(data[elemento].value_counts()) for elemento in data.columns]
    df = df.sort_values(by=['dtype', 'count_unique'])
    df = df.reset_index(drop=True)
    return df

###########################################################################################
'''              MODULO DE CLASES'''

#################################
"""Lista de objetos:
	- PandasTransform
	- PandasFeatureUnion
	- FeatureSelection
	- FeatureDivision
	- FeatureDiscretize
	- CorrSpearman
	- FeatureEncoder(nofunciona)
    - ColateColumn
    - BuscarIndex
    """
#################################


class PandasTransform(TransformerMixin, BaseEstimator):
 
    """Transforma un dataset en el mismo dataset con métodos de clase. https://zablo.net/blog/post/pandas-dataframe-in-scikit-learn-feature-union/index.html""";

    def __init__(self, fn):
        self.fn = fn

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, copy=None):
        return self.fn(X)

    
class PandasFeatureUnion(FeatureUnion):
    
    """
    Hace de FeatureUnion pero en Pandas
    """
    
    def fit_transform(self, X, y=None, **fit_params):
        self._validate_transformers()
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(
                transformer=trans,
                X=X,
                y=y,
                weight=weight,
                **fit_params)
            for name, trans, weight in self._iter())

        if not result:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs

    def merge_dataframes_by_column(self, Xs):
        return pd.concat(Xs, axis="columns", copy=False)

    def transform(self, X):
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(
                transformer=trans,
                X=X,
                y=None,
                weight=weight)
            for name, trans, weight in self._iter())
        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs

class FeatureSelection(BaseEstimator, TransformerMixin):

    """
    Recibe una lista de string con el nombre de las columnas que debe seleccionar, y transforma el dataframe conservando dichas columnas.
    """
    
    def __init__(self,selected_features):
        self.selected_features=selected_features

    def fit(self,X,y=None):
        return self
    
    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame): 
            return X[self.selected_features]
        else:
            print ('Error, no ingresaste un DataFrame')

class FeatureDivision(BaseEstimator, TransformerMixin):
    
    """
    Recibe dos labels, y te devuelve un dataframe con el cociente entre el primero por el segundo.
    """
    
    def __init__(self,columna_dividendo,columna_divisor):
        self.columna_dividendo=columna_dividendo
        self.columna_divisor=columna_divisor
        
    def fit(self,X,y=None):
        return self
    
    def transform(self, X, y=None):
        return pd.DataFrame(X[self.columna_dividendo]/X[self.columna_divisor],
                            index=X.index.values,
                            columns=[self.columna_dividendo+'_'+self.columna_divisor])


class FeatureDiscretize(BaseEstimator, TransformerMixin):
    
    """
    Discretiza las columnas ingresadas en la cantidade nbins especificados
    """
    
    def __init__(self,selected_features, nbins):
        self.selected_features=selected_features
        self.nbins = nbins
        
    def fit(self,X,y=None):
        return self
    
    def transform(self, X, y=None, nbins=10):
        return pd.DataFrame(pd.qcut(X[self.selected_features], self.nbins, labels=False))


class CorrSpearman(BaseEstimator, TransformerMixin):
    """
    Obtener el vector con los valores de correlación entre la variable target, y el resto de las features
    """
    def __init__(self, target):
        self.target = target
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return abs(X.corr(method='spearman')[self.target].drop(index=self.target).sort_values())
    
    
    
class FeatureEncoder(BaseEstimator, TransformerMixin):
    """
    NO FUNCIONA
    Aplica un LabelEncoder(), o un OneHotEncoder() en función del objeto que recibe
    """
    
#    def __init__(self):
#        pass

    def fit(self,X,y=None):
        return self
    
    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame): 
            try:
                return LabelEncoder().fit_transform(X)
            except:
                pass
            try:
                return OneHotEncoder().fit_transform(X)
            except:
                print ('error en el encoding de DataFrame')
        if isinstance(X, pd.Series):
            try:
                return LabelEncoder().fit_transform(X)
            except:
                print ('error en el encoding de Series')
        else:
            print ('Error en el metodo de clase')

class ColateColumn():
    """
    La columna "colate", debe entrar atras que la columna "label". Usamos un pd.reindex() 
    """
    
    def __init__(self, dataset, colate=0):
        self.dataset = dataset
        self.colate = colate
        
    def vos(self, colate):
        self.colate = colate
        return self
    
    def aca(self, aca):
        
        dicto = {}
        for i, elemento in enumerate(list(self.dataset.columns)):
            dicto[elemento] = i

        if dicto[aca] < dicto[self.colate]:
            n_colate = dicto[self.colate]
            n_aca = dicto[aca] + 1

            for i, elemento in enumerate(dicto.keys()):
                if dicto[elemento] >= n_aca and dicto[elemento] < n_colate:
                    dicto[elemento] += 1

            dicto[self.colate] = n_aca
            llaves = pd.Series(dicto).sort_values().index
            return self.dataset.reindex(llaves, axis=1)

class BuscarIndex():
        
    def __init__(self, dataset):
        self.dataset = dataset
        
        
    def cliente(self, cliente_id):
        dataset = self.dataset
        X_T = dataset.loc[dataset.index == cliente_id].T
        X_T.columns = X_T.iloc[0]
        X_T.drop(index = list(X_T.index)[0], inplace=True)
        X_T.index.name = f'| cliente: {cliente_id}'
        X_T.columns.name = 'Ventas Mensuales [$] | category:'
        return X_T
    
    def category(self, category):
        dataset = self.dataset
        X_T = dataset.loc[dataset.index == category]#.T
        #X_T.columns = X_T.iloc[0]
        X_T.set_index(list(dataset.columns)[0], drop=True, inplace=True)
        X_T.columns.name = 'Cantidad Vendida [Un.] | date:'
        return X_T.dropna()
            
#######################################################################################################


"""FUNCION DE AUTO-ML PARA OBTENER UNA TUPLA DE DICCIONARIOS HABIENDO IMPLENTADO ENSABLE"""

def evaluate_model(model_instance, X_train, y_train, X_test, y_test, gridSearch_params, gridSearch_bagging_params=False, random_state=np.random.randint(1000), n_split=3):
    
    # entreno el modelo default
    model_instance.fit(X_train, y_train)
    
    # calculo el score sobre los datos de test
    score_default_test = model_instance.score(X_test, y_test)
    
    # calculo la matriz de confusión
    predictions_default = model_instance.predict(X_test)
    confusion_matrix_default = metrics.confusion_matrix(y_test, predictions_default)
    
    ###################################################
    
    # gridSearch KFold:    
    cv_KFold = KFold(n_splits=n_split, shuffle=True)
    grid_search_CV_KFold_model = GridSearchCV(model_instance, gridSearch_params, n_jobs=-1, cv = cv_KFold)    
    grid_search_CV_KFold_model.fit(X_train, y_train)        
    scores_KFold = cross_val_score(model_instance, X_train, y_train, cv=cv_KFold, n_jobs=-1)
    #metelo en un dic
    mean_score_grid_search_CV_KFold_model = scores_KFold.mean()
    std_score_grid_search_CV_KFold_model = scores_KFold.std()
        
    score_grid_search_CV_KFold_model = grid_search_CV_KFold_model.best_score_
    params_grid_search_CV_KFold_model = grid_search_CV_KFold_model.best_estimator_.get_params()
    
    score_grid_search_CV_KFold_model_test = grid_search_CV_KFold_model.score(X_test, y_test)
    predictions_grid_search_CV_KFold_model = grid_search_CV_KFold_model.predict(X_test)    
    confusion_matrix_grid_search_CV_KFold_model = metrics.confusion_matrix(y_test, predictions_grid_search_CV_KFold_model)

    ###################################################
    
    # gridSearch Stratified KFold:    
    cv_Stratified_KFold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=random_state)
    grid_search_CV_Stratified_KFold_model = GridSearchCV(model_instance, gridSearch_params, n_jobs=-1, cv = cv_Stratified_KFold)    
    grid_search_CV_Stratified_KFold_model.fit(X_train, y_train)        
    scores_Stratified_KFold = cross_val_score(model_instance, X_train, y_train, cv=cv_Stratified_KFold, n_jobs=-1)
    
    ### dict
    mean_score_grid_search_CV_Stratified_KFold_model = scores_Stratified_KFold.mean()
    std_score_grid_search_CV_Stratified_KFold_model = scores_Stratified_KFold.std()    
    
    score_grid_search_CV_Stratified_KFold_model = grid_search_CV_Stratified_KFold_model.best_score_
    params_grid_search_CV_Stratified_KFold_model = grid_search_CV_Stratified_KFold_model.best_estimator_.get_params()
    
    score_grid_search_CV_Stratified_KFold_model_test = grid_search_CV_Stratified_KFold_model.score(X_test, y_test)
    predictions_grid_search_CV_Stratified_KFold_model = grid_search_CV_Stratified_KFold_model.predict(X_test)
    confusion_matrix_grid_search_CV_Stratified_KFold_model = metrics.confusion_matrix(y_test, predictions_grid_search_CV_Stratified_KFold_model)

    ###################################################
    
    if gridSearch_bagging_params:

        # bagging

        bagging_model_default = BaggingClassifier(base_estimator = model_instance)
        bagging_model_default.fit(X_train, y_train)
        score_bagging_model_default_test =  bagging_model_default.score(X_test, y_test)
        
        predictions_bagging_model_default = bagging_model_default.predict(X_test)
        confusion_matrix_bagging_model_default = metrics.confusion_matrix(y_test, predictions_bagging_model_default)    

        ###################################################

        # bagging Stratified KFold cross validation usando de base el mejor modelo de gridsearch estratificado
        base_estimator_stratified_grid_search = grid_search_CV_Stratified_KFold_model.best_estimator_
        cv_Stratified_KFold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=random_state)
        grid_search_bagging_model = GridSearchCV(BaggingClassifier(base_estimator = base_estimator_stratified_grid_search),
                               gridSearch_bagging_params, n_jobs=-1, cv = cv_Stratified_KFold)

        grid_search_bagging_model.fit(X_train, y_train)
        
        score_grid_search_bagging_model_test = grid_search_bagging_model.score(X_test, y_test)
        predictions_grid_search_bagging_model = grid_search_bagging_model.predict(X_test)
        confusion_matrix_grid_search_bagging_model = metrics.confusion_matrix(y_test, predictions_grid_search_bagging_model)
        scores_bagging_Stratified_KFold = cross_val_score(BaggingClassifier(base_estimator = base_estimator_stratified_grid_search),
                                                          X_train, y_train, cv=cv_Stratified_KFold, n_jobs=-1)
        mean_score_bagging_grid_search_CV_Stratified_KFold_model = scores_bagging_Stratified_KFold.mean()
        std_score_bagging_grid_search_CV_Stratified_KFold_model = scores_bagging_Stratified_KFold.std()    
        
        best_score_bagging_grid_search_CV_Stratified_KFold_model = grid_search_bagging_model.best_score_
        best_params_bagging_grid_search_CV_Stratified_KFold_model = grid_search_bagging_model.best_estimator_.get_params()
    
    else:
        
        score_bagging_model_default_test = None
        confusion_matrix_bagging_model_default = None
        mean_score_bagging_grid_search_CV_Stratified_KFold_model = None
        std_score_bagging_grid_search_CV_Stratified_KFold_model = None
        confusion_matrix_grid_search_bagging_model = None
        best_score_bagging_grid_search_CV_Stratified_KFold_model = None
        score_grid_search_bagging_model_test = None
        best_params_bagging_grid_search_CV_Stratified_KFold_model = None

                                                                                                                                     
    ###################################################
    
    
    # armo un diccionario con todos los valores de performance que calculé para los modelos
    result_score = {
        'default': {
            'test_score': score_default_test,
            'confusion_matrix': confusion_matrix_default            
        },
        'cv_kfold': {
            'mean_score_grid_search': mean_score_grid_search_CV_KFold_model,
            'std_score_grid_search': std_score_grid_search_CV_KFold_model,
            'model_score': score_grid_search_CV_KFold_model,
            'test_score': score_grid_search_CV_KFold_model_test,
            'confusion_matrix': confusion_matrix_grid_search_CV_KFold_model           
        },
        'cv_stratified_kfold': {
            'mean_score_grid_search': mean_score_grid_search_CV_Stratified_KFold_model,
            'std_score_grid_search': std_score_grid_search_CV_Stratified_KFold_model,
            'model_score': score_grid_search_CV_Stratified_KFold_model,
            'test_score': score_grid_search_CV_Stratified_KFold_model_test,
            'confusion_matrix': confusion_matrix_grid_search_CV_Stratified_KFold_model           
        },
        'bagging': {
            'test_score': score_bagging_model_default_test,
            'confusion_matrix': confusion_matrix_bagging_model_default           
        },
        'bagging_cv_stratified_kfold': {
            'mean_score_grid_search': mean_score_bagging_grid_search_CV_Stratified_KFold_model,
            'std_score_grid_search': std_score_bagging_grid_search_CV_Stratified_KFold_model,
            'model_score': best_score_bagging_grid_search_CV_Stratified_KFold_model,
            'test_score': score_grid_search_bagging_model_test,
            'confusion_matrix': confusion_matrix_grid_search_bagging_model           
        }
        
    }
    
    result_params = {
        'cv_kfold_best_estimator_params': [params_grid_search_CV_KFold_model],
        'cv_stratified_kfold_best_estimator_params': [params_grid_search_CV_Stratified_KFold_model],
        'bagging_cv_stratified_kfold_best_estimator_params': [best_params_bagging_grid_search_CV_Stratified_KFold_model]
    }
    
    for columna in result_params.keys():
        for elemento in result_params[columna][0].keys():
            result_params[columna][0][elemento] = [result_params[columna][0][elemento]]
    
    
    return result_score, result_params
    
    
    

def featureImportance(model, X_train):
    try:
        feature_importance = model.feature_importances_
    except:
        feature_importance = model.best_estimator_.feature_importances_
        
    feature_names = X_train.columns    
    importance = pd.DataFrame({'feature_names': feature_names, 'feature_importance': feature_importance})
    importance.sort_values(by = 'feature_importance', axis=0, ascending=False, inplace=True)
    return importance