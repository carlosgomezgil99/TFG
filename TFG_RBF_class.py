# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 18:35:01 2021

@author: carlo
"""

from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from tensorflow.keras.optimizers import RMSprop,Adam
from rbflayer import RBFLayer
from kmeans_initializer import InitCentersKMeans
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


#Carga de coordenadas una a una del archivo 'coordenadas.txt'
X=np.loadtxt('coordenadas.txt', delimiter=' ', skiprows=1,usecols=1)

Y=np.loadtxt('coordenadas.txt', delimiter=' ', skiprows=1,usecols=2)

Z=np.loadtxt('coordenadas.txt', delimiter=' ', skiprows=1,usecols=3)

v_X=np.loadtxt('coordenadas.txt', delimiter=' ', skiprows=1,usecols=4)

v_Y=np.loadtxt('coordenadas.txt', delimiter=' ', skiprows=1,usecols=5)

v_Z=np.loadtxt('coordenadas.txt', delimiter=' ', skiprows=1,usecols=6)



#Creación de matriz con coordenadas y respectivas velocidades en cada coordenada   
matriz=[]

for i in range(len(X)):
    a=[X[i],Y[i],Z[i],v_X[i],v_Y[i],v_Z[i]]

    matriz.append(a)

#Centro encontrado a partir de la media de la coordenada X
x0=1474538.05673
y0=-5811243.26336
z0= 2168958.84928

centro=[x0,y0,z0]
centro=np.asarray(centro)

#Encontrar 112 puntos más cercanos al centro
puntos=[]
diff=[]
matriz=np.asarray(matriz)
for p in matriz:
    dist=np.linalg.norm(abs(centro-p[:3]))
    diff.append(dist)
    if dist <=2300000:
        puntos.append(p)

#print(puntos,len(puntos))

puntos=np.asarray(puntos)

#Division de los 112 puntos en train y test (90 para train y 22 para test)
X_train, X_test, y_train, y_test = train_test_split(
                                        puntos[:,:3],
                                        puntos[:,3:],
                                        train_size   = 0.81,
                                        random_state = 1234,
                                        shuffle      = True
                                    )
#Normalización a la escala [0,1]
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled=scaler.transform(X_train)

X_test_scaled=scaler.transform(X_test)

scaler.fit(y_train)
y_train_scaled=scaler.transform(y_train)

y_test_scaled=scaler.transform(y_test)


#Creación del modelo de red RBF apoyada en los archivos rbflayer.py y kmeans_initializer.py
model = Sequential()
rbflayer = RBFLayer(80,
                        initializer=InitCentersKMeans(X_train_scaled),
                        betas=2,
                        input_shape=(3,))
model.add(rbflayer)
model.add(Dense(3, activation='linear'))

#Compilación del módelo
model.compile(loss='mean_squared_error',
                  optimizer='adam', metrics=['mse'])
print(model.summary())

#Entrenamiento del módelo
model.fit(X_train_scaled, y_train_scaled, epochs=800,batch_size=45,
          validation_data=(X_test_scaled,y_test_scaled),verbose=1)

#Evaluación del modelo
scores = model.evaluate(X_train_scaled, y_train_scaled,batch_size=90)
scores1 = model.evaluate(X_test_scaled,y_test_scaled)
print("train:", scores)
print("test:", scores1)

#Predicción de los valores reservados para test
predicts_scaled= model.predict(X_test_scaled)

#Desnormalización de los valores predichos
scaler.fit(y_train)
predicts=scaler.inverse_transform(predicts_scaled)

print("Predicciones: ",predicts)
print("Verdad: ",y_test)

score= mean_squared_error(y_test,predicts)
print("mse: ",score)

#Error coordenada a coordenada
diferencia=predicts-y_test
print(diferencia.tolist())

predicts_v=[]
y_test_v=[]
for i in range(predicts.shape[0]):
    predicts_v.append(np.linalg.norm(predicts[i]))
    y_test_v.append(np.linalg.norm(y_test[i]))

#Error total
diferencia_total=np.asarray(predicts_v)-np.asarray(y_test_v)
print(diferencia_total)
n_puntos=list(range(predicts.shape[0]))

#Gráfica de los errores totales
fig,ax=plt.subplots()
ax.plot(n_puntos,diferencia_total.tolist())
ax.set_title("Nº de puntos frente al error total")
ax.set_xlabel("Nº de puntos")
ax.set_ylabel("Error Total")
plt.show()

#Escritura de los errores coordenada a coordenada y totales en archivos .txt
with open('errores coordenada a coordenada RBF.txt', 'w') as f:
    for i in range(predicts.shape[0]):
        f.write(str(round(diferencia[i,0],3))+' '+
                str(round(diferencia[i,1],3))+' '+str(round(diferencia[i,2],3))+'\n')

with open('errores totales RBF.txt', 'w') as f:
    for i in range(predicts.shape[0]):
        f.write(str(round(diferencia_total[i],3))+'\n')

#Media y desviación típica de los errores totales
print("media: ", round(np.mean(diferencia_total),3))
print("desviación típica: ", round(np.std(diferencia_total),3))
