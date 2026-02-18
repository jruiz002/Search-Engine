from typing import Optional, Set
import heapq
from domain import Node, State

try:
    from .smart_problem import SmartMazeProblem
except ImportError:
    from smart_problem import SmartMazeProblem

class SmartAStar:
    
    def __init__(self):
        self.nodes_explored = 0
        self.nodes_generated = 0
        self.max_frontier_size = 0
    
    def search(self, problem: SmartMazeProblem, verbose: bool = False) -> Optional[Node]:
        # Reiniciar estadísticas
        self.nodes_explored = 0
        self.nodes_generated = 0
        self.max_frontier_size = 0
        
        # Nodo inicial
        start_node = Node(problem.initial_state())
        start_node.heuristic_cost = problem.heuristic(start_node.state)
        self.nodes_generated += 1
        
        if problem.is_goal(start_node.state):
            return start_node
        
        # Frontera (cola de prioridad)
        frontier = []
        heapq.heappush(frontier, (start_node.total_cost, self.nodes_generated, start_node))
        
        explored: Set[State] = set()
        
        # Mapeo de estados a nodos en frontera para optimización
        frontier_states = {start_node.state: start_node}
        
        if verbose:
            print("Iniciando búsqueda A* inteligente...")
            print(f"Estado inicial: {start_node.state}")
            print(f"Objetivos: {problem.goal_positions}")
        
        while frontier:
            # Actualizar estadística de tamaño máximo de frontera
            self.max_frontier_size = max(self.max_frontier_size, len(frontier))
            
            # Obtener nodo con menor f(n) = g(n) + h(n)
            _, _, node = heapq.heappop(frontier)
            
            # Remover de frontier_states si aún está ahí
            if node.state in frontier_states and frontier_states[node.state] == node:
                del frontier_states[node.state]
            
            # Verificar si llegamos al objetivo
            if problem.is_goal(node.state):
                if verbose:
                    self._print_search_stats()
                    self._print_solution_stats(node, problem)
                return node
            
            # Marcar como explorado
            explored.add(node.state)
            self.nodes_explored += 1
            
            if verbose and self.nodes_explored % 100 == 0:
                print(f"Nodos explorados: {self.nodes_explored}, Frontera: {len(frontier)}, "
                      f"Costo actual: {node.path_cost:.2f}")
            
            # Expandir vecinos
            for action in problem.actions(node.state):
                child_state = problem.result(node.state, action)
                
                # Saltar si ya fue explorado
                if child_state in explored:
                    continue
                
                # Calcular costo del paso usando la red neuronal
                step_cost = problem.step_cost(node.state, action, child_state)
                
                # Calcular costos del nodo hijo
                path_cost = node.path_cost + step_cost
                heuristic_cost = problem.heuristic(child_state)
                
                # Crear nodo hijo
                child = Node(child_state, node, action, path_cost, heuristic_cost)
                self.nodes_generated += 1
                
                # Verificar si ya está en frontera con mejor costo
                if child_state in frontier_states:
                    existing_node = frontier_states[child_state]
                    if child.total_cost < existing_node.total_cost:
                        # Encontramos un camino mejor, actualizar
                        frontier_states[child_state] = child
                        heapq.heappush(frontier, (child.total_cost, self.nodes_generated, child))
                else:
                    # Agregar nuevo nodo a la frontera
                    frontier_states[child_state] = child
                    heapq.heappush(frontier, (child.total_cost, self.nodes_generated, child))
        
        if verbose:
            print("No se encontró solución")
            self._print_search_stats()
        
        return None
    
    def _print_search_stats(self):
        print("\n" + "="*50)
        print("ESTADÍSTICAS DE BÚSQUEDA A* INTELIGENTE")
        print("="*50)
        print(f"Nodos explorados: {self.nodes_explored:,}")
        print(f"Nodos generados: {self.nodes_generated:,}")
        print(f"Tamaño máximo de frontera: {self.max_frontier_size:,}")
        print(f"Factor de ramificación efectivo: {self.nodes_generated/max(1, self.nodes_explored):.2f}")
        print("="*50)
    
    def _print_solution_stats(self, solution_node: Node, problem: SmartMazeProblem):
        print("\n" + "="*50)
        print("SOLUCIÓN ENCONTRADA")
        print("="*50)
        print(f"Costo total del camino: {solution_node.path_cost:.2f}")
        print(f"Longitud del camino: {self._get_path_length(solution_node)}")
        
        path_summary = problem.get_path_costs_summary(solution_node)
        if path_summary:
            print(f"Costo promedio por paso: {path_summary['average_cost_per_step']:.2f}")
            print("\nMateriales encontrados en el camino:")
            for material, count in sorted(path_summary['materials_encountered'].items()):
                cost = problem.color_classifier.cost_mapping.get(material, 0)
                print(f"  {material}: {count} pasos (costo {cost})")
        
        print("="*50)
    
    def _get_path_length(self, solution_node: Node) -> int:
        length = 0
        node = solution_node
        while node.parent is not None:
            length += 1
            node = node.parent
        return length
    
    def get_path(self, solution_node: Node) -> list:
        if not solution_node:
            return []
        
        path = []
        node = solution_node
        while node is not None:
            path.append(node.state)
            node = node.parent
        
        path.reverse()
        return path
    
    def analyze_path_materials(self, solution_node: Node, problem: SmartMazeProblem) -> dict:
        if not solution_node:
            return {}
        
        path_states = self.get_path(solution_node)
        analysis = {
            'path_length': len(path_states),
            'total_cost': solution_node.path_cost,
            'materials': [],
            'cost_breakdown': {},
            'efficiency_metrics': {}
        }
        
        total_material_cost = 0.0
        
        # Analizar cada paso del camino
        for i, state in enumerate(path_states):
            r, c = state.key
            
            try:
                # Obtener color RGB del tile
                avg_rgb = problem.color_classifier.get_average_rgb_from_tile(
                    problem.env.image, r, c, problem.env.tile_size
                )
                
                # Predecir material y costo
                material = problem.color_classifier.predict_color(avg_rgb)
                cost = problem.color_classifier.predict_cost(avg_rgb)
                
                analysis['materials'].append({
                    'step': i,
                    'position': (r, c),
                    'rgb': avg_rgb,
                    'material': material,
                    'cost': cost
                })
                
                # Acumular costos por material
                if material not in analysis['cost_breakdown']:
                    analysis['cost_breakdown'][material] = {'count': 0, 'total_cost': 0.0}
                
                analysis['cost_breakdown'][material]['count'] += 1
                analysis['cost_breakdown'][material]['total_cost'] += cost
                total_material_cost += cost
                
            except Exception as e:
                print(f"Error analizando posición {(r, c)}: {e}")
        
        # Calcular métricas de eficiencia
        if len(path_states) > 0:
            analysis['efficiency_metrics'] = {
                'average_cost_per_step': total_material_cost / len(path_states),
                'cost_variance': self._calculate_cost_variance(analysis['materials']),
                'most_expensive_material': max(analysis['cost_breakdown'].items(), 
                                              key=lambda x: x[1]['total_cost'])[0] if analysis['cost_breakdown'] else None,
                'most_common_material': max(analysis['cost_breakdown'].items(), 
                                           key=lambda x: x[1]['count'])[0] if analysis['cost_breakdown'] else None
            }
        
        return analysis
    
    def _calculate_cost_variance(self, materials: list) -> float:
        if not materials:
            return 0.0
        
        costs = [m['cost'] for m in materials]
        mean_cost = sum(costs) / len(costs)
        variance = sum((cost - mean_cost) ** 2 for cost in costs) / len(costs)
        return variance