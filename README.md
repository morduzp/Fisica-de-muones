# Análisis del decaimiento de muones

Este repositorio contiene un script en **Python** para el análisis del tiempo de vida de los muones a partir de datos experimentales almacenados en un archivo de texto (`muones.txt`).  
El programa implementa el procesamiento de los datos, la construcción de histogramas de sobrevivientes, y la aplicación de ajustes exponenciales con y sin corrección de fondo, con el fin de estimar parámetros físicos relevantes.

## Objetivos
- Determinar el tiempo de vida medio del muón \(\tau\).  
- Estimar la constante de Fermi \(G_F\) a partir del valor obtenido de \(\tau\).  
- Calcular la razón de abundancia de muones positivos y negativos \(\rho = N^+/N^-\).  

## Limitación en GitHub
El archivo original `muones.txt` no pudo ser cargado en este repositorio debido a que excede el límite estándar de GitHub (100 MB).  
Por esta razón, aquí se encuentra únicamente el código de análisis. Para ejecutar el programa, es necesario contar con un archivo de datos local en el mismo formato (una columna con tiempos en **nanosegundos**).

## Instrucciones de uso
1. Clone este repositorio.  
2. Copie su archivo `muones.txt` en el directorio raíz.  
3. Ejecute el script con:

```bash
python analisis_muones.py
