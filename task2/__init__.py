# Task 2 - Red Neuronal y Algoritmos Inteligentes
# 
# Este paquete implementa:
# - Red neuronal MLP desde cero (neural_network.py)
# - Clasificador de colores RGB (color_classifier.py) 
# - Problema de laberinto con costos dinámicos (smart_problem.py)
# - Algoritmo A* inteligente (smart_search.py)
# - Punto de entrada principal (main_task2.py)

from .neural_network import MLP
from .color_classifier import ColorClassifier
from .smart_problem import SmartMazeProblem, SmartMazeState
from .smart_search import SmartAStar

__all__ = ['MLP', 'ColorClassifier', 'SmartMazeProblem', 'SmartMazeState', 'SmartAStar']