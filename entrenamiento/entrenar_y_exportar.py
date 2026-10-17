# -*- coding: utf-8 -*-
"""Entrena la CNN de perros y gatos y la exporta a TensorFlow.js.

Misma arquitectura que el cuaderno de clase "Perros y gatos --modelo CNN--".

Local: python entrenar_y_exportar.py [epocas_maximas]
"""
#---------------------------------------------------------------------
# 1) LIBRERÍAS
#---------------------------------------------------------------------
import os     #Para manejar carpetas y rutas de archivos
import sys    #Para leer argumentos de la terminal (sys.argv)

import cv2                           #OpenCV: para cambiar tamaño y pasar a blanco y negro
import numpy as np                   #Arreglos numéricos
import tensorflow as tf              #La librería de redes neuronales (incluye Keras)
import tensorflow_datasets as tfds   #Para descargar el dataset de perros y gatos
import tensorflowjs as tfjs          #Para convertir el modelo y usarlo en una página web

#---------------------------------------------------------------------
# 2) CONFIGURACIÓN
#---------------------------------------------------------------------
#Épocas MÁXIMAS (en clase se usaron 50). Si la red deja de mejorar antes, se detiene sola (ver EarlyStopping)
EPOCAS = int(sys.argv[1]) if len(sys.argv) > 1 else 50
TAMANO_LOTE = 32
TAMANO_IMG = 100   #Las fotos se reducen a 100x100 píxeles

#Carpeta donde se guarda el modelo: perros_gatos_web/modelo
CARPETA_SALIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "modelo")


#---------------------------------------------------------------------
# 3) DATOS
#---------------------------------------------------------------------
#Tensorflow datasets tiene error al descargar el set de perros y gatos.
#Esta línea (igual que en el cuaderno de clase) le pone la dirección correcta de descarga
setattr(tfds.image_classification.cats_vs_dogs, '_URL',
        "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip")

#Descargar el set de datos de perros y gatos (~23,000 fotos; etiqueta 0 = gato, 1 = perro)
datos, metadatos = tfds.load('cats_vs_dogs', as_supervised=True, with_info=True)

#Cada foto tiene un tamaño distinto y viene a color.
#Las pasamos todas a 100x100 y a blanco y negro (igual que en clase)
X = []  #imagenes de entrada (pixeles)
y = []  #etiquetas (perro o gato)
for imagen, etiqueta in datos['train']:
  imagen = cv2.resize(imagen.numpy(), (TAMANO_IMG, TAMANO_IMG))
  imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
  X.append(imagen.reshape(TAMANO_IMG, TAMANO_IMG, 1))  #Cambiar tamano a 100,100,1
  y.append(etiqueta.numpy())

#Normalizar: pixeles de 0-255 a 0-1 (float32 en vez de float para usar la mitad de memoria)
X = np.array(X).astype(np.float32) / 255
y = np.array(y)
print("Datos listos:", X.shape, "| gatos:", int((y == 0).sum()), "| perros:", int((y == 1).sum()))


#---------------------------------------------------------------------
# 4) MODELO CNN (igual al cuaderno "Perros y gatos modelo CNN")
#---------------------------------------------------------------------
#Tamaño en cada paso: 100 -> 98 -> 49 -> 47 -> 23 -> 21 -> 10  (10x10x128 = 12,800 al aplanar)
modelo = tf.keras.Sequential([
  #Entrada: foto de 100x100 en blanco y negro (1 canal)
  tf.keras.layers.Input(shape=(TAMANO_IMG, TAMANO_IMG, 1)),

  tf.keras.layers.Conv2D(32, (3,3), activation='relu'),   #Rasgos simples: bordes, texturas
  tf.keras.layers.MaxPooling2D(2, 2),                     #Reduce a la mitad

  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),   #Formas: ojos, orejas
  tf.keras.layers.MaxPooling2D(2, 2),

  tf.keras.layers.Conv2D(128, (3,3), activation='relu'),  #Partes completas: hocico, cara
  tf.keras.layers.MaxPooling2D(2, 2),

  tf.keras.layers.Flatten(),                              #Aplana a un vector de 12,800
  tf.keras.layers.Dense(100, activation='relu'),
  #Salida: UNA neurona con sigmoide -> número entre 0 (gato) y 1 (perro)
  tf.keras.layers.Dense(1, activation='sigmoid')
])

#binary_crossentropy: función de error para clasificar en DOS clases (gato / perro)
modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
modelo.summary()


#---------------------------------------------------------------------
# 5) ENTRENAR
#---------------------------------------------------------------------
#En clase esta CNN se sobreajustaba: llegaba a 99% en entrenamiento pero se quedaba en ~83%
#en validación, y su error de validación empezaba a SUBIR desde la época ~5.
#EarlyStopping vigila el error de validación: si no mejora en 5 épocas, detiene el entrenamiento
#y regresa a los pesos de la MEJOR época. Así exportamos el mejor modelo, no el sobreajustado.
detener = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

#validation_split=0.15 -> el 15% de las fotos NO se usan para entrenar, solo para medir
historial = modelo.fit(X, y, batch_size=TAMANO_LOTE, validation_split=0.15,
                       epochs=EPOCAS, callbacks=[detener], verbose=2)

mejor = int(np.argmin(historial.history['val_loss']))
print(f"Mejor época: {mejor + 1} | "
      f"precisión entrenamiento {historial.history['accuracy'][mejor]:.4f} | "
      f"precisión validación {historial.history['val_accuracy'][mejor]:.4f}")


#---------------------------------------------------------------------
# 6) EXPORTAR a TensorFlow.js -> modelo/model.json + .bin
#---------------------------------------------------------------------
os.makedirs(CARPETA_SALIDA, exist_ok=True)
tfjs.converters.save_keras_model(modelo, CARPETA_SALIDA)
print(f"Exportado a {os.path.abspath(CARPETA_SALIDA)}")
