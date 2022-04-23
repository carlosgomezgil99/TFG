# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 15:17:22 2021

@author: carlo

"""

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
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



#Creación del modelo con una capa oculta y 60 neuronas ocultas
model = Sequential()
model.add(Dense(60,input_shape=(3,) ,activation='tanh',   kernel_initializer='he_normal',
                bias_initializer='he_normal'))

model.add(Dense(3, activation='sigmoid'))

#Compilación del modelo
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mse'])
print(model.summary())

#Entrenamiento del modelo
model.fit(X_train_scaled, y_train_scaled, epochs=1000,batch_size=45,
          validation_data=(X_test_scaled,y_test_scaled), verbose=1)

#Evaluación del modelo
scores = model.evaluate(X_train_scaled, y_train_scaled,batch_size=90)
scores1 = model.evaluate(X_test_scaled,y_test_scaled)
print("train:", scores)
print("test:", scores1)

#Predicción de los datos de test
predicts_scaled= model.predict(X_test_scaled)

scaler.fit(y_train)
predicts=scaler.inverse_transform(predicts_scaled)

print("Predicciones: ",predicts)
print("Verdad: ",y_test)

#MSE de la salida deseada y la predicha
score= mean_squared_error(y_test,predicts)
print("mse: ",score)

#Diferencia de velocidad coordenada a coordenada
diferencia=predicts-y_test
print(diferencia.tolist())

predicts_v=[]
y_test_v=[]
for i in range(predicts.shape[0]):
    predicts_v.append(np.linalg.norm(predicts[i]))
    y_test_v.append(np.linalg.norm(y_test[i]))

#Diferencia de velocidad total entre vectores
diferencia_total=np.asarray(predicts_v)-np.asarray(y_test_v)
print(diferencia_total)
n_puntos=list(range(1,23))



#Escritura de las diferencias de velocidades coordenada a coordenada y total en archivos .txt
with open('errores coordenada a coordenada MLP.txt', 'w') as f:
    for i in range(predicts.shape[0]):
        f.write(str(round(diferencia[i,0],3))+' '+
                str(round(diferencia[i,1],3))+' '+str(round(diferencia[i,2],3))+'\n')

with open('errores totales MLP.txt', 'w') as f:
    for i in range(predicts.shape[0]):
        f.write(str(round(diferencia_total[i],3))+'\n')


#Carga de errores
"""
e_x_mlp=np.loadtxt('errores coordenada a coordenada MLP.txt',delimiter='\t',usecols=0)
e_y_mlp=np.loadtxt('errores coordenada a coordenada MLP.txt',delimiter='\t',usecols=1)
e_z_mlp=np.loadtxt('errores coordenada a coordenada MLP.txt',delimiter='\t',usecols=2)
"""
e_mlp=np.loadtxt('errores totales MLP.txt', delimiter='\n',usecols=0)
"""
e_x_rbf=np.loadtxt('errores coordenada a coordenada RBF.txt',delimiter=' ',usecols=0)
e_y_rbf=np.loadtxt('errores coordenada a coordenada RBF.txt',delimiter=' ',usecols=1)
e_z_rbf=np.loadtxt('errores coordenada a coordenada RBF.txt',delimiter=' ',usecols=2)
"""
e_rbf=np.loadtxt('errores totales RBF.txt', delimiter='\n',usecols=0)

#Media y desviación tpica de ambas redes componente a componente y totales
"""
print(e_x_mlp.mean(),e_x_mlp.std())
print(e_y_mlp.mean(),e_y_mlp.std())
print(e_z_mlp.mean(),e_z_mlp.std())

print(e_mlp.mean(),e_mlp.std())

print(e_x_rbf.mean(),e_x_rbf.std())
print(e_y_rbf.mean(),e_y_rbf.std())
print(e_z_rbf.mean(),e_z_rbf.std())

print(e_rbf.mean(),e_rbf.std())
"""
#Gráfica comparativa entre diferencia de velocidad total de ambas redes
fig,ax=plt.subplots()
ax.plot(n_puntos,e_mlp,linestyle = 'dashed')
ax.plot(n_puntos,e_rbf, color='red')
ax.set_xlabel("Nº de punto")
ax.set_ylabel("Diferencia velocidad total (m/a)")
plt.show()



#[2.30470340e+06, -4.87481716e+06,  3.39518699e+06]