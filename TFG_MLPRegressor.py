# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 16:47:04 2022

@author: carlo
"""

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor

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

scaler.fit(X_test)
X_test_scaled=scaler.transform(X_test)

scaler.fit(y_train)
y_train_scaled=scaler.transform(y_train)

scaler.fit(y_test)
y_test_scaled=scaler.transform(y_test)

#Modelo regresor con 2 capas ocultas de 50 neuronas cada capa y con función tanh
model=MLPRegressor(hidden_layer_sizes=(50,50),activation='tanh',
                   random_state=1, max_iter=400).fit(X_train_scaled, y_train_scaled)

predicts_scaled=model.predict(X_test_scaled)
scaler.fit(y_test)
predicts=scaler.inverse_transform(predicts_scaled)

print("Predicciones: ",predicts)

print("Verdad: ",y_test)
print(model.score(X_test_scaled,y_test_scaled))

score= mean_squared_error(y_test,predicts)
print("mse: ",score)

#Error coordenada a coordenada
diferencia=predicts-y_test
print(diferencia)

predicts_v=[]
y_test_v=[]
for i in range(predicts.shape[0]):
    predicts_v.append(np.linalg.norm(predicts[i]))
    y_test_v.append(np.linalg.norm(y_test[i]))
#Error total
diferencia_total=np.asarray(predicts_v)-np.asarray(y_test_v)
print(diferencia_total)
n_puntos=list(range(predicts.shape[0]))

#Gráfica de errores
fig,ax=plt.subplots()
ax.plot(n_puntos,diferencia_total.tolist())
ax.set_title("Nº de puntos frente al error total")
ax.set_xlabel("Nº de puntos")
ax.set_ylabel("Error Total")
plt.show()

#Escritura de los errores coordenada a coordenada y totales en archivos .txt
with open('errores coordenada a coordenada MLP.txt', 'w') as f:
    for i in range(predicts.shape[0]):
        f.write(str(round(diferencia[i,0],3))+' '+
                str(round(diferencia[i,1],3))+' '+str(round(diferencia[i,2],3))+'\n')

with open('errores totales MLP.txt', 'w') as f:
    for i in range(predicts.shape[0]):
        f.write(str(round(diferencia_total[i],3))+'\n')

#Media y desviación típica del error total
print("media: ", round(np.mean(diferencia_total),3))
print("desviación típica: ", round(np.std(diferencia_total),3))