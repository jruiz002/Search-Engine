from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Optional

class State:
    """
    Representa un estado en el problema de búsqueda.
    """
    def __init__(self, key: Any):
        self.key = key

    def __eq__(self, other):
        return isinstance(other, State) and self.key == other.key

    def __hash__(self):
        return hash(self.key)

    def __str__(self):
        return str(self.key)

class Action:
    """
    Representa una acción que se puede tomar en un estado.
    """
    def __init__(self, name: str, cost: float = 1.0):
        self.name = name
        self.cost = cost

    def __str__(self):
        return self.name

class Problem(ABC):
    """
    Clase base abstracta para un problema de búsqueda.
    """

    @abstractmethod
    def initial_state(self) -> State:
        """Devuelve el estado inicial del problema."""
        pass

    @abstractmethod
    def is_goal(self, state: State) -> bool:
        """Devuelve True si el estado dado es un estado objetivo."""
        pass

    @abstractmethod
    def actions(self, state: State) -> List[Action]:
        """Devuelve una lista de acciones aplicables en el estado dado."""
        pass

    @abstractmethod
    def result(self, state: State, action: Action) -> State:
        """Devuelve el estado que resulta de ejecutar la acción dada en el estado dado."""
        pass

    @abstractmethod
    def step_cost(self, state: State, action: Action, next_state: State) -> float:
        """Devuelve el costo de tomar una acción para llegar al siguiente estado."""
        pass
    
    @abstractmethod
    def heuristic(self, state: State) -> float:
        """Devuelve el costo estimado para llegar al objetivo desde el estado dado."""
        return 0.0

class Node:
    """
    Representa un nodo en el árbol de búsqueda.
    """
    def __init__(self, state: State, parent=None, action: Action=None, path_cost: float=0.0, heuristic_cost: float=0.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.heuristic_cost = heuristic_cost # h(n)
        
    @property
    def total_cost(self) -> float:
        """f(n) = g(n) + h(n)"""
        return self.path_cost + self.heuristic_cost

    def __lt__(self, other):
        return self.total_cost < other.total_cost
