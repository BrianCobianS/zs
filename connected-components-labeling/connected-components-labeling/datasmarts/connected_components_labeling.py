# Importamos las librerías de interés.
import argparse
import cv2
import numpy as np
from skimage import measure  # El paquete measure contiene la implementación del algoritmo.

# Definimos los parámetros de entrada que, en este caso, es sólo uno: -i, el cual usamos para definir la ruta
# de la imagen sobre la que operaremos.
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-i', '--image', required=True, type=str, help='Ruta a la imagen de entrada.')
arguments = vars(argument_parser.parse_args())

plate = cv2.imread(arguments['image'])

# Extraemos el canal "(V)alue" de la imagen convertida a espacio HSV, para luego aplicar un "adaptive threshold" o umbral
# (https://datasmarts.net/es/primeros-pasos-en-opencv-parte-18/)
# adaptativo, con el fin de revelar los caracteres en la matrícula.
hsv_plate = cv2.cvtColor(plate, cv2.COLOR_BGR2HSV)
V = cv2.split(hsv_plate)[2]
thresholded = cv2.adaptiveThreshold(V, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 17, 3)

# Mostramos la imagen original y la binaria.
cv2.imshow('Matrícula', plate)
cv2.imshow('Imagen binaria', thresholded)

# Aquí es donde llevamos a cabo el análisis de componentes conectadas sobre la imagen binaria. También inicializamos
# una máscara que contendrá sólo las regiones más grandes, las cuales son las que nos interesan.
#
# El método measure.label hace todo el trabajo pesado. El primer parámetro es la imagen binaria, el segundo, connectivity=2,
# indica que aplicaremos conectividad 8, mientras que background=0 le dice a scikit-image que los pixeles con dicho valor
# corresponden al fondo y, por tanto, han de ser ignorados.
# El arreglo labels tiene las mismas dimensiones que la imagen de entrada, sólo que cada píxel estará etiquetado con
# el número correspondiente a la región a la que pertenece.
labels = measure.label(thresholded, connectivity=2, background=0)
mask = np.zeros(thresholded.shape, dtype='uint8')
print(f'Se encontraron {len(np.unique(labels))} regiones.')

# Ciclamos sobre cada etiqueta.
for i, label in enumerate(np.unique(labels)):
    if label == 0:
        print('Etiqueta: 0 (fondo)')
        continue

    # Construimos una máscara para mostrar sólo las componentes conectadas de la etiqueta actual.
    print(f'Etiqueta: {i}')
    label_mask = np.zeros(thresholded.shape, dtype='uint8')
    label_mask[labels == label] = 255
    num_pixels = cv2.countNonZero(label_mask)

    # Si el número de píxeles en la componente es suficientemente larga, probablemente sea una región de interés,
    # por lo que la agregamos a nuestra máscara final.
    if 300 < num_pixels < 1500:
        mask = cv2.add(mask, label_mask)

    # Mostramos la máscara de *esta* etiqueta.
    cv2.imshow('Etiqueta', label_mask)
    cv2.waitKey(0)

# Mostramos las componentes más grandes (es decir, regiones de interés).
cv2.imshow('Regiones de interés', mask)
cv2.waitKey(0)
