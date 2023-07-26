import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasRegressor

base = pd.read_csv('autos.csv', encoding= 'ISO-8859-1')

# tratamento de dados ------------------------------------------------------

# axis= 1    apagar todos os elementos
base = base.drop('dateCrawled', axis= 1)
base = base.drop('dateCreated', axis= 1)
base = base.drop('nrOfPictures', axis= 1)
base = base.drop('postalCode', axis= 1)
base = base.drop('lastSeen', axis= 1)
base = base.drop('name', axis= 1)
base = base.drop('seller', axis= 1)
base = base.drop('offerType', axis= 1)

# base['name'].value_counts()

# variavel para os valores inconsistentes
# loc = localizacao de registros
# valores muito baixos
i1 = base.loc[base.price <= 10]

# media dos preços
#base.price.mean()

# só se manterão na base os valores maiores que 10
base = base[base.price > 10]

# os valores abaixo de 350k se manterão
base = base.loc[base.price < 350000]

valores = {'vehicleType': 'limousine', 'gearbox': 'manuell',
           'model': 'golf', 'fuelType': 'benzin',
           'noRepairDamage': 'nein'}

# irá remover os dados com NaN dos valores separados acima
base = base.fillna(value = valores)

previsores = base.iloc[:, 1:13].values
preco_real = base.iloc[:, 0].values

# criacao de dummys - valores categoricos passam a ser numéricos (enconde 
# dos atributos categoricos)
# exemplo tipo em (iris_simples.py)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_previsores = LabelEncoder()

previsores[:, 0] = labelencoder_previsores.fit_transform(previsores[:, 0])
previsores[:, 1] = labelencoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 3] = labelencoder_previsores.fit_transform(previsores[:, 3])
previsores[:, 5] = labelencoder_previsores.fit_transform(previsores[:, 5])
previsores[:, 8] = labelencoder_previsores.fit_transform(previsores[:, 8])
previsores[:, 9] = labelencoder_previsores.fit_transform(previsores[:, 9])
previsores[:, 10] = labelencoder_previsores.fit_transform(previsores[:, 10])

# 0 - 0 0 0
# 2 - 0 1 0
# 3 - 0 0 1

onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0,1,3,5,8,9,10])],remainder='passthrough') #atualizado
previsores = onehotencoder.fit_transform(previsores).toarray()

# fim do tratamento de dados -----------------------------------------------

# Construção da rede neural

def criar_rede():
    regressor = Sequential()

    # 1ª camada oculta
    regressor.add(Dense(units= 158, activation= 'relu', input_dim= 316))

    # 2ª camada oculta
    regressor.add(Dense(units= 158, activation= 'relu'))

    # camada de saida
    # apenas uma saída pois só quero saber o preço do produto
    # funcao linear pois só queremos o valor do preco, nao queremos boolean
    # nem probabilidade
    regressor.add(Dense(units= 1, activation= 'linear'))

    regressor.compile(loss= 'mean_absolute_error', optimizer= 'adam',
                      metrics = ['mean_absolute_error'])
    return regressor

regressor = KerasRegressor(build_fn= criar_rede,
                           epochs= 100,
                           batch_size= 300)
resultados = cross_val_score(estimator= regressor,
                             X= previsores, y= preco_real,
                             cv= 10, scoring= 'mean_absolute_error')
media = resultados.mean()

desvio = resultados.std()