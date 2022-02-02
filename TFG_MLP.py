# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 15:17:22 2021

@author: carlo

"""
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from keras.models import Sequential
from keras.layers import Dense, Flatten
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from IPython.display import clear_output
from tensorflow.keras import callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
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



# 'relu',kernel_initializer='he_normal', PReLU, LeakyReLU, swish,'selu', kernel_initializer='lecun_normal'

earlystop=callbacks.EarlyStopping(monitor='loss',mode='min',
                                   patience=5,restore_best_weights=True)

model = Sequential()
model.add(Dense(50,input_shape=(3,) ,activation='tanh',   kernel_initializer='he_normal',
                bias_initializer='he_normal'))
#model.add(Dense(50, activation='relu',kernel_initializer='he_normal',
   #bias_initializer='he_normal'))
#model.add(Dense(30, activation='relu',kernel_initializer='he_normal',
    #bias_initializer='he_normal'))
#model.add(Dense(300, activation='relu', kernel_initializer='random_normal',bias_initializer='zeros'))
#model.add(Dense(300, activation='relu', kernel_initializer='random_normal',bias_initializer='zeros'))
#model.add(Dense(250, activation='tanh'))
model.add(Dense(3, activation='sigmoid'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mse'])
print(model.summary())
model.fit(X_train_scaled, y_train_scaled, epochs=1000,batch_size=45,validation_data=(X_test_scaled,y_test_scaled), verbose=1)
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

with open('errores coordenada a coordenada MLP.txt', 'w') as f:
    for i in range(predicts.shape[0]):
        f.write(str(round(diferencia[i,0],3))+' '+
                str(round(diferencia[i,1],3))+' '+str(round(diferencia[i,2],3))+'\n')

with open('errores totales MLP.txt', 'w') as f:
    for i in range(predicts.shape[0]):
        f.write(str(round(diferencia_total[i],3))+'\n')
"""


error=[]
for i in range(len(salida_ver)):
    error.append(abs(predicts[i]-salida_ver[i]))
print('error: ',error)

print(round(np.array(predicts_normal,float).mean(),3))
print(round(np.array(predicts_normal,float).std(),3))

"""
#i=17/22/25, relu  