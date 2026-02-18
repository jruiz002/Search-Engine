from domain import Problem, State, Action
from environment import Environment
import math
import os

try:
    from .color_classifier import ColorClassifier
except ImportError:
    from color_classifier import ColorClassifier

class SmartMazeState(State):
    
    def __init__(self, r: int, c: int):
        super().__init__((r, c))
        self.r = r
        self.c = c

    def __lt__(self, other):
        return (self.r, self.c) < (other.r, other.c)


class SmartMazeProblem(Problem):
    def __init__(self, environment: Environment, color_classifier: ColorClassifier):
        self.env = environment
        self.color_classifier = color_classifier
        
        # Verificar que el clasificador esté entrenado
        if not color_classifier.is_trained:
            raise ValueError("El clasificador de colores debe estar entrenado antes de usar SmartMazeProblem")
        
        # Encontrar posición de inicio
        start_pos = self.env.get_start()
        if start_pos is None:
            raise ValueError("No se encontró la posición de inicio (píxel rojo)")
        
        self.start_r, self.start_c = start_pos
        self.goal_positions = set(self.env.get_goals())
        
        if not self.goal_positions:
            print("Advertencia: No se encontraron objetivos en el laberinto")
    
    def initial_state(self) -> State:
        return SmartMazeState(self.start_r, self.start_c)
    
    def is_goal(self, state: State) -> bool:
        return state.key in self.goal_positions
    
    def actions(self, state: State) -> list[Action]:
        actions = []
        r, c = state.key
        rows, cols = self.env.get_grid_size()
        
        # Movimientos posibles: Arriba, Abajo, Izquierda, Derecha
        moves = [
            ("UP", -1, 0),
            ("DOWN", 1, 0),
            ("LEFT", 0, -1),
            ("RIGHT", 0, 1)
        ]
        
        for name, dr, dc in moves:
            nr, nc = r + dr, c + dc
            
            # Verificar límites
            if 0 <= nr < rows and 0 <= nc < cols:
                # Verificar si NO es un obstáculo sólido
                if not self.env.is_obstacle(nr, nc):
                    # El costo se calculará dinámicamente en step_cost
                    actions.append(Action(name, cost=1.0))  # Costo base, se ajustará después
        
        return actions
    
    def result(self, state: State, action: Action) -> State:
        r, c = state.key
        
        if action.name == "UP":
            return SmartMazeState(r - 1, c)
        elif action.name == "DOWN":
            return SmartMazeState(r + 1, c)
        elif action.name == "LEFT":
            return SmartMazeState(r, c - 1)
        elif action.name == "RIGHT":
            return SmartMazeState(r, c + 1)
        
        return state
    
    def step_cost(self, state: State, action: Action, next_state: State) -> float:
        nr, nc = next_state.key
        
        # Verificar si el siguiente estado es un obstáculo sólido
        if self.env.is_obstacle(nr, nc):
            return 999.0  # Costo muy alto para obstáculos
        
        try:
            # Obtener color RGB promedio del tile destino
            avg_rgb = self.color_classifier.get_average_rgb_from_tile(
                self.env.image, nr, nc, self.env.tile_size
            )
            
            # Inferencia en vivo: pasar RGB por la red neuronal
            predicted_cost = self.color_classifier.predict_cost(avg_rgb)
            
            return predicted_cost
            
        except Exception as e:
            # En caso de error, usar costo por defecto
            print(f"Error en step_cost: {e}")
            return 5.0  # Costo moderado por defecto
    
    def heuristic(self, state: State) -> float:
        if not self.goal_positions:
            return 0.0
        
        r, c = state.key
        min_dist = float('inf')
        
        for gr, gc in self.goal_positions:
            dist = abs(r - gr) + abs(c - gc)
            if dist < min_dist:
                min_dist = dist
        
        # Multiplicar por el costo mínimo para mantener admisibilidad
        min_cost = min(self.color_classifier.cost_mapping.values())
        return min_dist * min_cost
    
    def get_path_costs_summary(self, solution_node) -> dict:
        if not solution_node:
            return {}
        
        # Reconstruir el camino
        path = []
        node = solution_node
        while node.parent is not None:
            path.append((node.state.key, node.path_cost - node.parent.path_cost))
            node = node.parent
        path.reverse()
        
        # Analizar materiales encontrados
        materials_count = {}
        total_cost = 0.0
        
        for (r, c), step_cost in path:
            try:
                avg_rgb = self.color_classifier.get_average_rgb_from_tile(
                    self.env.image, r, c, self.env.tile_size
                )
                material = self.color_classifier.predict_color(avg_rgb)
                materials_count[material] = materials_count.get(material, 0) + 1
                total_cost += step_cost
            except:
                pass
        
        return {
            'path_length': len(path),
            'total_cost': solution_node.path_cost,
            'materials_encountered': materials_count,
            'average_cost_per_step': total_cost / len(path) if path else 0
        }