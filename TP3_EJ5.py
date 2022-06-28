import numpy as np
import openpyxl as xls
import matplotlib.pyplot as plt

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

def inicializar_pesos(n_entrada, n_capa_2, n_capa_3):
    randomgen = np.random.default_rng()
    #n_entrada=NEURONAS_ENTRADA, n_capa_2=NEURONAS_CAPA_OCULTA, n_capa_3=numero_clases
    w1 = 0.1 * randomgen.standard_normal((n_entrada, n_capa_2))
    b1 = 0.1 * randomgen.standard_normal((1, n_capa_2))

    w2 = 0.1 * randomgen.standard_normal((n_capa_2, n_capa_3))
    b2 = 0.1 * randomgen.standard_normal((1,n_capa_3))

    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}

def ejecutar_adelante(x, pesos):
    # Funcion de entrada (a.k.a. "regla de propagacion") para la primera capa oculta
    z = x.dot(pesos["w1"]) + pesos["b1"]

    # Funcion de activacion ReLU para la capa oculta (h -> "hidden")
    h = np.maximum(0, z)

    # Salida de la red (funcion de activacion lineal). Esto incluye la salida de todas
    # las neuronas y para todos los ejemplos proporcionados
    y = h.dot(pesos["w2"]) + pesos["b2"]
    return {"z": z, "h": h, "y": y}

def clasificar(x, pesos):
    # Corremos la red "hacia adelante"
    resultados_feed_forward = ejecutar_adelante(x, pesos)
    
    # Buscamos la(s) clase(s) con scores mas altos (en caso de que haya mas de una con 
    # el mismo score estas podrian ser varias). Dado que se puede ejecutar en batch (x 
    # podria contener varios ejemplos), buscamos los maximos a lo largo del axis=1 
    # (es decir, por filas)
    max_scores = np.argmax(resultados_feed_forward["y"], axis=1)
    # Tomamos el primero de los maximos (podria usarse otro criterio, como ser eleccion aleatoria)
    # Nuevamente, dado que max_scores puede contener varios renglones (uno por cada ejemplo),
    # retornamos la primera columna
    try:
        return max_scores[:, 0]
    except:
        return max_scores[:]

def train(x, t, pesos, learning_rate, epochs, n_validacion, x_val, t_val):
    # Cantidad de filas (i.e. cantidad de ejemplos)
    m = np.size(x, 0) 
    Loss_ant = 1000
    Loss_act= 0
    valores_loss_training=[]
    valores_loss_validacion=[]

    for i in range(epochs):
        # Ejecucion de la red hacia adelante
        resultados_feed_forward = ejecutar_adelante(x, pesos)
        y = resultados_feed_forward["y"]
        h = resultados_feed_forward["h"]
        z = resultados_feed_forward["z"]

        # Calculo de la funcion de perdida global con MSE.
        err=np.zeros((m,1))
        for j in range(m):
            err[j]=(t[j]-y[j])
        loss=(np.sum(err**2))/m

        # Extraemos los pesos a variables locales
        w1 = pesos["w1"]
        b1 = pesos["b1"]
        w2 = pesos["w2"]
        b2 = pesos["b2"]

        # Ajustamos los pesos: Backpropagation
        dL_dy = -2*err/m                # Para todas las salidas
        dL_dw2 = h.T.dot(dL_dy)                         # Ajuste para w2
        dL_db2 = np.sum(dL_dy, axis=0, keepdims=True)   # Ajuste para b2

        dL_dh = dL_dy.dot(w2.T)
        
        dL_dz = dL_dh             # El calculo dL/dz = dL/dh * dh/dz. La funcion "h" es la funcion de activacion de la capa oculta,
        dL_dz[z <= 0] = 0         # para la que usamos ReLU. La derivada de la funcion ReLU: 1(z > 0) (0 en otro caso)
        # dh_dz = h.T.dot((1-h))  # Sigmoide //     dL/dz = dL/dh * sigma(x)*(1-sigma(x))
        # dL_dz = dL_dh.dot(dh_dz)# Sigmoide

        dL_dw1 = x.T.dot(dL_dz)                         # Ajuste para w1
        dL_db1 = np.sum(dL_dz, axis=0, keepdims=True)   # Ajuste para b1

        # Aplicamos el ajuste a los pesos
        w1 += -learning_rate * dL_dw1
        b1 += -learning_rate * dL_db1
        w2 += -learning_rate * dL_dw2
        b2 += -learning_rate * dL_db2

        # Actualizamos la estructura de pesos
        # Extraemos los pesos a variables locales
        pesos["w1"] = w1
        pesos["b1"] = b1
        pesos["w2"] = w2
        pesos["b2"] = b2

        # Parada temprana por validación
        if i %n_validacion ==0:
            if i>5:     #Debo esperar que converja ya que al principio puede oscilar
                #Overfitting
                Loss_act=validar(x_val,t_val,pesos)
                if Loss_act <= Loss_ant:
                    Loss_ant=Loss_act
                elif Loss_act > (Loss_ant * 1.5): 
                    print("Entrenamiento detenido por Overfitting (oscilación mayor al 50%) en epoch", i,"\n")
                    break
                #Correlacion -> Diferencia del 80% entre Training y Validacion
                error_val=np.abs(loss-Loss_act)/loss
                if (error_val> 0.8):
                    print("Entrenamiento detenido en epoch", i,"\n")
                    print("No Correlación (diferencia mayor al 80%)\n")
                    #break
                elif (error_val> 0.2):
                    print("No Correlación (diferencia mayor al 20%)\n")
                print("\nEpoch", i,"\n", "Training Loss: ",loss," Validation Loss: ", Loss_act, " Correlation Error: ",error_val)
        valores_loss_training.append(float(loss))
        valores_loss_validacion.append(float(Loss_act))
    
    #Gráfico
    plt.figure(1)
    plt.title("VALOR DE PÉRDIDA")
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.plot(range(epochs),valores_loss_training)
    plt.plot(range(epochs),valores_loss_validacion)
    plt.grid()
    plt.show()
        
def iniciar(numero_clases, numero_ejemplos, graficar_datos, aleatoriedad_train, aleatoriedad_test, aleatoriedad_val):
    # Generamos conjuntos de train, validacion y test
    dataset=generador_datos(7,17379)
    lim=int(17380*0.8)
    lim2=int(17380*0.9)

    x = dataset[0:lim,0:6]
    t = dataset[0:lim,6]
    
    x_test= dataset[lim:lim2,0:6]
    t_test = dataset[lim:lim2,6]
    x_val= dataset[lim2:17379,0:6]
    t_val = dataset[lim2:17379,6]

    # Inicializa pesos de la red
    NEURONAS_CAPA_OCULTA = 100
    NEURONAS_ENTRADA = 6
    pesos = inicializar_pesos(n_entrada=NEURONAS_ENTRADA, n_capa_2=NEURONAS_CAPA_OCULTA, n_capa_3=numero_clases)

    # Entrena
    LEARNING_RATE=0.445
    EPOCHS=1000
    N_VALIDACION=EPOCHS/10
    train(x, t, pesos, LEARNING_RATE, EPOCHS, N_VALIDACION, x_val, t_val)
    
    # Error de test
    print("Error del test:",validar(x_test,t_test,pesos))

def validar(xt,tt,pesos):
    m=len(xt)
    validacion_feed_forward = ejecutar_adelante(xt, pesos)
    y_val= validacion_feed_forward["y"]
    err_val=np.zeros((m,1))
    for j in range(m):
        err_val[j]=(tt[j]-y_val[j])
    loss_val=(np.sum(err_val**2))/m
    return loss_val

iniciar(numero_clases=1, numero_ejemplos=300, graficar_datos=True, aleatoriedad_train=0.1, aleatoriedad_test=0.15, aleatoriedad_val=0.3)