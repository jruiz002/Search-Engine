from typing import List, Set, Deque, Optional
from collections import deque
import heapq
from domain import Problem, Node, State, Action

class SearchAlgorithm:
    def search(self, problem: Problem) -> Optional[Node]:
        raise NotImplementedError

class BFS(SearchAlgorithm):
    """
    Implementación de Búsqueda en Anchura (BFS).
    """
    def search(self, problem: Problem) -> Optional[Node]:
        node = Node(problem.initial_state())
        if problem.is_goal(node.state):
            return node
        
        frontier: Deque[Node] = Deque([node])
        explored: Set[State] = set()
        
        while frontier:
            node = frontier.popleft() # FIFO
            explored.add(node.state)
            
            for action in problem.actions(node.state):
                child_state = problem.result(node.state, action)
                if child_state not in explored and all(n.state != child_state for n in frontier):
                    child = Node(child_state, node, action, node.path_cost + problem.step_cost(node.state, action, child_state))
                    if problem.is_goal(child.state):
                        return child
                    frontier.append(child)
        return None

class DFS(SearchAlgorithm):
    """
    Implementación de Búsqueda en Profundidad (DFS).
    """
    def search(self, problem: Problem) -> Optional[Node]:
        node = Node(problem.initial_state())
        if problem.is_goal(node.state):
            return node
        
        frontier: List[Node] = [node]
        explored: Set[State] = set()
        
        while frontier:
            node = frontier.pop() # LIFO
            explored.add(node.state)
            
            if problem.is_goal(node.state):
                return node

            for action in problem.actions(node.state):
                child_state = problem.result(node.state, action)
                if child_state not in explored and all(n.state != child_state for n in frontier):
                    child = Node(child_state, node, action, node.path_cost + problem.step_cost(node.state, action, child_state))
                    frontier.append(child)
        return None

class AStar(SearchAlgorithm):
    """
    Implementación de Búsqueda A* (A Star).
    """
    def search(self, problem: Problem) -> Optional[Node]:
        start_node = Node(problem.initial_state())
        start_node.heuristic_cost = problem.heuristic(start_node.state)
        
        if problem.is_goal(start_node.state):
            return start_node
            
        frontier = [] # Cola de prioridad
        heapq.heappush(frontier, (start_node.total_cost, start_node))
        explored: Set[State] = set()
        
        while frontier:
            _, node = heapq.heappop(frontier)
            
            if problem.is_goal(node.state):
                return node
            
            explored.add(node.state)
            
            for action in problem.actions(node.state):
                child_state = problem.result(node.state, action)
                if child_state in explored:
                    continue
                    
                path_cost = node.path_cost + problem.step_cost(node.state, action, child_state)
                heuristic = problem.heuristic(child_state)
                child = Node(child_state, node, action, path_cost, heuristic)
                
                # Verificar si el hijo está en la frontera con un costo mayor
                in_frontier = False
                for i, (p, n) in enumerate(frontier):
                    if n.state == child_state:
                        in_frontier = True
                        if child.total_cost < p:
                            frontier[i] = (child.total_cost, child)
                            heapq.heapify(frontier) # Reordenar
                        break
                
                if not in_frontier:
                    heapq.heappush(frontier, (child.total_cost, child))
                    
        return None
