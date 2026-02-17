# Buscador de Caminos en Laberintos (Maze Search Engine)

Este proyecto implementa un motor de búsqueda que encuentra caminos óptimos en laberintos representados por imágenes. Utiliza algoritmos de Inteligencia Artificial clásica para navegar desde un punto de inicio hasta una meta, evitando obstáculos.

## Características

- **Algoritmos de Búsqueda**: Implementa BFS (Búsqueda en Anchura), DFS (Búsqueda en Profundidad) y A* (A Star) para encontrar soluciones.
- **Entrada Visual**: Procesa imágenes (PNG, BMP, JPG) donde los colores representan el entorno.
- **Visualización de Resultados**: Genera imágenes de salida mostrando el camino encontrado sobre el laberinto original o una cuadrícula discretizada.

## Requisitos del Sistema

Para ejecutar este programa en otra computadora, necesitas tener instalado:

1.  **Python 3.x**: El lenguaje de programación base.
2.  **Librerías de Python**:
    -   `numpy`: Para manejo de matrices y cálculos numéricos.
    -   `Pillow` (PIL): Para procesamiento de imágenes.

## Instalación

1.  **Descargar el código**: Clona este repositorio o descarga los archivos en una carpeta de tu computadora.
2.  **Instalar dependencias**: Abre una terminal o línea de comandos, navega a la carpeta del proyecto y ejecuta el siguiente comando:

    ```bash
    pip install numpy Pillow
    ```

## Cómo Usar

1.  **Preparar las Imágenes**:
    -   Coloca las imágenes de tus laberintos en la carpeta llamada `assets` dentro del directorio del proyecto.
    -   **Formato de Colores**:
        -   🟥 **Rojo**: Punto de Inicio.
        -   🟩 **Verde**: Meta / Objetivo.
        -   ⬛ **Negro**: Paredes / Obstáculos.
        -   ⬜ **Blanco**: Camino libre.

2.  **Ejecutar el Programa**:
    -   Desde la terminal, ejecuta el archivo principal:

    ```bash
    python main.py
    ```

3.  **Ver los Resultados**:
    -   El programa creará una carpeta llamada `output`.
    -   Dentro de `output`, encontrarás subcarpetas para cada imagen procesada con las soluciones visualizadas para cada algoritmo (BFS, DFS, A*).

## Estructura del Proyecto

- `main.py`: Punto de entrada del programa.
- `domain.py`: Define las estructuras básicas (Estado, Acción, Nodo, Problema).
- `environment.py`: Procesa la imagen y la convierte en una cuadrícula lógica.
- `problem.py`: Define las reglas específicas del problema del laberinto.
- `search.py`: Contiene la implementación de los algoritmos de búsqueda.
- `visualization.py`: Herramientas para dibujar los caminos encontrados en las imágenes.
- `assets/`: Carpeta para las imágenes de entrada.
- `output/`: Carpeta para las imágenes de salida.
