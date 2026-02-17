from domain import Problem, State, Action
from environment import Environment
import math

class MazeState(State):
    def __init__(self, r: int, c: int):
        super().__init__((r, c))
        self.r = r
        self.c = c

    def __lt__(self, other):
        """Para un orden consistente si es necesario"""
        return (self.r, self.c) < (other.r, other.c)

class MazeProblem(Problem):
    def __init__(self, environment: Environment):
        self.env = environment
        start_pos = self.env.get_start()
        # Asegurarse de que se encuentre start_pos, de lo contrario volver a (0,0) o generar error
        if start_pos is None:
             raise ValueError("No se encontró la posición de inicio (píxel rojo)")
        
        self.start_r, self.start_c = start_pos
        self.goal_positions = set(self.env.get_goals())
        
        if not self.goal_positions:
             # ¿Advertencia o Error? Dado que la Tarea 1.1 dice "Identificar Objetivos", esperamos al menos uno.
             pass

    def initial_state(self) -> State:
        return MazeState(self.start_r, self.start_c)

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
            if 0 <= nr < rows and 0 <= nc < cols:
                # Verificar si hay obstáculo
                if not self.env.is_obstacle(nr, nc):
                    actions.append(Action(name, cost=1.0))
        
        return actions

    def result(self, state: State, action: Action) -> State:
        r, c = state.key
        if action.name == "UP":
            return MazeState(r - 1, c)
        elif action.name == "DOWN":
            return MazeState(r + 1, c)
        elif action.name == "LEFT":
            return MazeState(r, c - 1)
        elif action.name == "RIGHT":
            return MazeState(r, c + 1)
        return state

    def step_cost(self, state: State, action: Action, next_state: State) -> float:
        return action.cost

    def heuristic(self, state: State) -> float:
        """
        Distancia Manhattan al objetivo más cercano.
        """
        if not self.goal_positions:
            return 0.0
        
        r, c = state.key
        min_dist = float('inf')
        
        for gr, gc in self.goal_positions:
            dist = abs(r - gr) + abs(c - gc)
            if dist < min_dist:
                min_dist = dist
        
        return min_dist
