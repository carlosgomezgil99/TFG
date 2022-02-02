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
from rbflayer import RBFLayer, InitCentersRandom
from kmeans_initializer import InitCentersKMeans
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
scaler = MinMaxScaler()
X=np.loadtxt('coordenadas.txt', delimiter=' ', skiprows=1,usecols=1)

Y=np.loadtxt('coordenadas.txt', delimiter=' ', skiprows=1,usecols=2)

Z=np.loadtxt('coordenadas.txt', delimiter=' ', skiprows=1,usecols=3)

v_X=np.loadtxt('coordenadas.txt', delimiter=' ', skiprows=1,usecols=4)

v_Y=np.loadtxt('coordenadas.txt', delimiter=' ', skiprows=1,usecols=5)

v_Z=np.loadtxt('coordenadas.txt', delimiter=' ', skiprows=1,usecols=6)



    
matriz=[]


for i in range(len(X)):
    a=[X[i],Y[i],Z[i],v_X[i],v_Y[i],v_Z[i]]

    matriz.append(a)


x0=1474538.05673
y0=-5811243.26336
z0= 2168958.84928

centro=[x0,y0,z0]
centro=np.asarray(centro)

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

X_train, X_test, y_train, y_test = train_test_split(
                                        puntos[:,:3],
                                        puntos[:,3:],
                                        train_size   = 0.81,
                                        random_state = 1234,
                                        shuffle      = True
                                    )


scaler.fit(X_train)
X_train_scaled=scaler.transform(X_train)

scaler.fit(X_test)
X_test_scaled=scaler.transform(X_test)

scaler.fit(y_train)
y_train_scaled=scaler.transform(y_train)

scaler.fit(y_test)
y_test_scaled=scaler.transform(y_test)
"""
n=30
km=KMeans(n_clusters=n,init='random',
                       n_init=1, random_state=0, max_iter=1000)
km.fit(X_train_scaled)
centers=km.cluster_centers_
dist=[]
for c1 in centers:
    for c2 in centers:
        diff=np.linalg.norm(abs(c1-c2))
        dist.append(diff)

max_dist=max(dist)
sigma=max_dist/sqrt(2*n)
beta=(1/sigma)**2
"""
model = Sequential()
rbflayer = RBFLayer(80,
                        initializer=InitCentersKMeans(X_train_scaled),
                        betas=2,
                        input_shape=(3,))
model.add(rbflayer)
model.add(Dense(3, activation='linear'))

model.compile(loss='mean_squared_error',
                  optimizer='adam', metrics=['mse'])
print(model.summary())
model.fit(X_train_scaled, y_train_scaled, epochs=800,batch_size=90,
          validation_data=(X_test_scaled,y_test_scaled),verbose=1)
scores = model.evaluate(X_train_scaled, y_train_scaled,batch_size=90)
scores1 = model.evaluate(X_test_scaled,y_test_scaled)
print("train:", scores)
print("test:", scores1)
predicts_scaled= model.predict(X_test_scaled)

scaler.fit(y_test)
predicts=scaler.inverse_transform(predicts_scaled)

print("Predicciones: ",predicts)
print("Verdad: ",y_test)

score= mean_squared_error(y_test,predicts)
print("mse: ",score)

diferencia=predicts-y_test
print(diferencia.tolist())

predicts_v=[]
y_test_v=[]
for i in range(predicts.shape[0]):
    predicts_v.append(np.linalg.norm(predicts[i]))
    y_test_v.append(np.linalg.norm(y_test[i]))

diferencia_total=np.asarray(predicts_v)-np.asarray(y_test_v)
print(diferencia_total)
n_puntos=list(range(predicts.shape[0]))

fig,ax=plt.subplots()
ax.plot(n_puntos,diferencia_total.tolist())
ax.set_title("Nº de puntos frente al error total")
ax.set_xlabel("Nº de puntos")
ax.set_ylabel("Error Total")
plt.show()

with open('errores coordenada a coordenada RBF.txt', 'w') as f:
    for i in range(predicts.shape[0]):
        f.write(str(round(diferencia[i,0],3))+' '+
                str(round(diferencia[i,1],3))+' '+str(round(diferencia[i,2],3))+'\n')

with open('errores totales RBF.txt', 'w') as f:
    for i in range(predicts.shape[0]):
        f.write(str(round(diferencia_total[i],3))+'\n')
