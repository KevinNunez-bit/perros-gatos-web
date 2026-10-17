# ¿Perro o Gato? Red neuronal convolucional (CNN)

Proyecto de la materia **Redes Neuronales**, Universidad de Sonora.

Abre la cámara del celular (o sube una foto) y una red convolucional entrenada con el dataset *cats_vs_dogs* (~23,000 fotos) dice si es un perro o un gato.

- **Arquitectura:** 3 capas Conv2D + MaxPooling (32, 64, 128 filtros) → Dense(100) → Dense(1, sigmoide)
- **Entrada:** foto de 100×100 en blanco y negro
- **Precisión en validación:** ~85% (con EarlyStopping para evitar sobreajuste)

El modelo se entrenó en Python (TensorFlow/Keras), se exportó a TensorFlow.js y corre directamente en el navegador del celular.

- `index.html`: la página
- `modelo/`: el modelo exportado (`model.json` + pesos `.bin`)
- `entrenamiento/entrenar_y_exportar.py`: código de entrenamiento y exportación
- `iniciar.sh`: enciende el servidor y un túnel de ngrok, y muestra el link y un QR
