import os
import sys
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import Environment
from visualization import Visualizer
from task2.color_classifier import ColorClassifier
from task2.smart_problem import SmartMazeProblem
from task2.smart_search import SmartAStar



def train_color_classifier():
    # Inicializar clasificador  
    classifier = ColorClassifier()
    
    # Mostrar mapeo de costos
    classifier.print_cost_mapping()
    
    # Entrenar modelo
    results = classifier.train_model(
        test_size=0.2,
        epochs=150,
        batch_size=64,
        hidden_sizes=[64, 32],
        learning_rate=0.01
    )
    
    # Guardar modelo entrenado
    classifier.save_model("trained_color_model.pkl")
    
    return classifier, results

def run_smart_pathfinding(classifier, maze_image_path, output_dir):
    print(f"\n" + "="*60)
    print("BÚSQUEDA INTELIGENTE A* CON COSTOS DINÁMICOS")
    print("="*60)
    
    # Configurar entorno
    tile_size = 20
    env = Environment(maze_image_path, tile_size)
    
    # Crear problema inteligente
    smart_problem = SmartMazeProblem(env, classifier)
    
    # Ejecutar A* inteligente
    smart_astar = SmartAStar()
    solution = smart_astar.search(smart_problem, verbose=True)
    
    if solution:
        print(f"\n¡SOLUCIÓN ENCONTRADA!")
        
        # Calcular longitud del camino
        path = []
        current = solution
        while current is not None:
            path.append((current.state.r, current.state.c))
            current = current.parent
        path.reverse()
        
        print(f"- Longitud: {len(path)} pasos")
        print(f"- Costo total: {solution.path_cost:.2f}")
        
        # Guardar visualización simple del camino
        visualizer = Visualizer(env)
        output_path = os.path.join(output_dir, "smart_astar_solution.png")
        visualizer.draw_path_on_original(solution, output_path)
        print(f"Camino guardado en: {output_path}")
        
        return solution
    else:
        print("No se encontró solución")
        return None

def demonstrate_intelligent_pathfinding():

    print("\n" + "-"*80)
    print("TASK 2: RED NEURONAL PARA NAVEGACIÓN INTELIGENTE")
    print("-"*80)
    
    # Crear directorio de salida
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # STEP 1: Entrenar clasificador de colores
    try:
        classifier, training_results = train_color_classifier()
    except Exception as e:
        print(f"Error en entrenamiento: {e}")
        return
    
    user_maze_path = "assets/laberinto.png"
        
    # STEP 3: Ejecutar búsqueda inteligente
    try:
        solution = run_smart_pathfinding(classifier, user_maze_path, output_dir)

            
    except Exception as e:
        print(f"Error en búsqueda inteligente: {e}")
        import traceback
        traceback.print_exc()

def main():
    demonstrate_intelligent_pathfinding()

if __name__ == "__main__":
    main()