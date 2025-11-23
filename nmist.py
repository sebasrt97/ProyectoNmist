import tensorflow as tf
import numpy as np
#dividimos los datos de entrenamiento para obtener un 20% de datos de validación
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping

# 1. Extracción y Preparación de datos
(xtrain, ytrain), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalizacion de los datos (convertir de 0-255 a 0-1)
xtrain= xtrain/255
x_test= x_test/255

ytrain=ytrain.astype(np.int16)
y_test=y_test.astype(np.int16)


#Como vamos a usar un early_stopping hacemos una separacion del xtrain e ytrain y valores este es el crupier, tambien se podria hacer dentro del modelo pero no tendira tanta aletoriedad que buscmaods
x_train,x_val,y_train,y_val = train_test_split(xtrain, ytrain, test_size=0.16, random_state=42)

####se podira pasar las salidas de y_train e y_test como un array pero nos obligaria a un paso mas tanto aqui como el cambio del loss en el compile
# y_train=to_categorical(y_Train)

#No es lo mismo
# x_train[2]
# x_train.shape[1]
# x_train.shape
##output: (50400, 28 ,28) que son las imagenes, ancho y alto

## Estructura de las neuronas:

num_clases = 10
tf.random.set_seed(42)
input_shape=(x_train.shape[1],x_train.shape[2])

#Buena practica: maneajr el input_shape fuera creando una variable para ello y luego pasarlo por parametro

model = Sequential([
    #Acuerdate que se le pasa solo el numero del ancho y el alto, se ahce accediendo por el shape
    Flatten(input_shape=input_shape),
    #Flatten(input_shpae=(28,28)), -> otra opcion sabiendo que es un 28 * 28
    Dense(255, activation="relu"),
    Dense(120, activation="relu"),
    Dense(10, activation="softmax")
])

##Configuracion del modelom las reglas para empezar

opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss="sparse_categorical_crossentropy",
              optimizer= opt,
              metrics=["accuracy"]
)

## Ejecucion del entramienot, utilizando un early_stopping
from keras import callbacks

early_stopping=EarlyStopping(monitor='val_accuracy', mode="max", restore_best_weights=True, patience=4)

modelaje = model.fit(
    x_train,
    y_train,
    epochs=20,
    validation_data=(x_val,y_val),
    verbose=0,
    callbacks=[early_stopping]

)

## Comprobracion de la clasificacion

from sklearn.metrics import classification_report,confusion_matrix

predicciones=model.predict(x_test)
#se pide al modelo que prediga las eituqetas para el x_test
# un ejeemplo primera salida : [0.001, 0.003, 0.015, 0.001, 0.001, 0.000, 0.001, 0.970, 0.005, 0.003']
#                                 0      1      2      3      4      5      6      7      8       9

predicciones=np.argmax(predicciones,axis=1)
#con npo.argmax buscamos el valor de la equita con mayor probailidad

informe=classification_report(y_test, predicciones, target_names=['0','1','2','3','4','5','6','7','8','9'])
#Aqu la funciona importada genera el informe comparando con el modelo de predcciones, las etiquetas de y_test: la verdad absoluta, y por ultimo los argumentos donde se veran los nombres de las clases

print(informe)