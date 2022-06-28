import numpy as np
import openpyxl as xls
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

def generador_datos(cant_atributos,end):
    file = xls.load_workbook('Bike_Sharing_Dataset.xlsx', data_only=True)
    set=file["Bike_Sharing_Dataset"]
    datos=np.zeros((end, cant_atributos))
    i=0
    for row in set.rows:
        j=0
        for columna in row:
            if j<cant_atributos and i<end:
                datos[i][j]=set.cell(i+2,j+2).value
            j+=1   
        i+=1
    # Centrado y Normalización, para que los resultados sean comparables (Es un escalamiento y un centrado)
    datos_cen = datos - datos.mean(axis=0) #resto el promedio de columna
    datos_norm = datos_cen / datos_cen.max()
    file.close()
    return datos_norm

dataset=generador_datos(7,17379)
x=dataset[:,0:6]
t=dataset[:,6]


#Tipo de red neuronal, en este caso secuencial
red= Sequential() 

#Cantidad de Neuronas de la Primera capa oculta, Cantidad de Variables de Entrada, Función de Activación
red.add(Dense(30, input_dim=6, activation='relu')) 
red.add(Dense(30)) #Nueva Capa Oculta
red.add(Dense(30)) 
red.add(Dense(30)) 
red.add(Dense(30)) 
red.add(Dense(30)) 
red.add(Dense(30)) 
red.add(Dense(1)) #se define 1 salida

#FUNCION DE PERDIDA
red.compile(loss='mean_squared_error', optimizer='adam') 

#ENTRENAMIENTO 20% de los ejemplos sean usados para validacion y el numero de epochs
history=red.fit(x,t,validation_split=0.2,epochs=50) 
scores=red.evaluate(x,t)

#GRAFICA DE LOSS
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Funcion de Perdida')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training','Validacion'])
plt.show()