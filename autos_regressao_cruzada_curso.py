import pandas as pd
import tensorflow as tf # atualizado: tensorflow==2.0.0-beta1
from tensorflow.keras.models import Sequential # atualizado: tensorflow==2.0.0-beta1
from tensorflow.keras.layers import Dense # atualizado: tensorflow==2.0.0-beta1
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras import backend as k # atualizado: tensorflow==2.0.0-beta1
from sklearn import metrics
import time

inicio = time.time()
base = pd.read_csv('autos.csv', encoding = 'ISO-8859-1')

base = base.drop('dateCrawled', axis = 1)
base = base.drop('dateCreated', axis = 1)
base = base.drop('nrOfPictures', axis = 1)
base = base.drop('postalCode', axis = 1)
base = base.drop('lastSeen', axis = 1)
base = base.drop('name', axis = 1)
base = base.drop('seller', axis = 1)
base = base.drop('offerType', axis = 1)

base = base[base.price > 10]
base = base.loc[base.price < 350000]

valores = {'vehicleType': 'limousine', 'gearbox': 'manuell',
           'model': 'golf', 'fuelType': 'benzin',
           'notRepairedDamage': 'nein'}
base = base.fillna(value = valores)

previsores = base.iloc[:, 1:13].values
preco_real = base.iloc[:, 0].values

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0,1,3,5,8,9,10])],remainder='passthrough')
previsores = onehotencoder.fit_transform(previsores).toarray()


def criar_rede(): # atualizado: tensorflow==2.0.0-beta1
        k.clear_session()
        regressor = Sequential([
        tf.keras.layers.Dense(units=158, activation = 'relu', input_dim=316),
        tf.keras.layers.Dense(units=158, activation = 'relu'),
        tf.keras.layers.Dense(units=1, activation = 'linear')])
        regressor.compile(loss = 'mean_absolute_error', optimizer = 'adam',
                      metrics = ['mean_absolute_error'])
        return regressor

regressor = KerasRegressor(build_fn = criar_rede,
                           epochs = 100,
                           batch_size = 300)
resultados = cross_val_score(estimator = regressor,
                             X = previsores, y = preco_real,
                             cv = 10, scoring = 'neg_mean_absolute_error')
fim = time.time()
media = resultados.mean()
desvio = resultados.std()
print(fim - inicio)
        