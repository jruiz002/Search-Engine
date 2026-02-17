from PIL import Image, ImageDraw
from environment import Environment
from domain import Node

class Visualizer:
    def __init__(self, environment: Environment):
        self.env = environment

    def draw_path_on_original(self, node: Node, output_path: str):
        """
        Dibuja el camino en la imagen original (línea roja conectando centros).
        Corresponde a la visualización de la Tarea 1.2.
        """
        if node is None:
            print("No path found to visualize on original image.")
            return

        # Reconstruir camino
        path_nodes = []
        curr = node
        while curr:
            path_nodes.append(curr.state.key)
            curr = curr.parent
        path_nodes.reverse() # Inicio a Meta

        # Cargar imagen original
        img = self.env.image.copy()
        draw = ImageDraw.Draw(img)
        
        tile_size = self.env.tile_size
        offset = tile_size // 2
        
        # Convertir coords de cuadrícula a coords de píxel (centro del mosaico)
        points = []
        for r, c in path_nodes:
            x = c * tile_size + offset
            y = r * tile_size + offset
            points.append((x, y))
        
        # Dibujar línea
        if len(points) > 1:
            draw.line(points, fill="blue", width=3)
            
        img.save(output_path)
        print(f"Path visualized on original image saved to {output_path}")

    def draw_path_on_grid(self, node: Node, output_path: str):
        """
        Dibuja el camino en la cuadrícula discretizada.
        Corresponde a la visualización de la Tarea 1.3.
        """
        if node is None:
            print("No path found to visualize on grid.")
            return

        # Reconstruir camino
        path_nodes = set()
        curr = node
        while curr:
            path_nodes.add(curr.state.key)
            curr = curr.parent

        # Crear imagen de cuadrícula
        rows, cols = self.env.get_grid_size()
        tile_size = self.env.tile_size
        out_img = Image.new('RGB', (cols * tile_size, rows * tile_size), 'white')
        pixels = out_img.load()
        
        # Colores
        COLOR_WALL = (0, 0, 0)
        COLOR_FREE = (255, 255, 255)
        COLOR_START = (255, 0, 0)
        COLOR_GOAL = (0, 255, 0)
        COLOR_PATH = (0, 0, 255) # Camino azul
        
        start_pos = self.env.get_start()
        goals = set(self.env.get_goals())
        
        for r in range(rows):
            for c in range(cols):
                color = COLOR_FREE
                
                # Determinar tipo de celda
                if self.env.is_obstacle(r, c):
                    color = COLOR_WALL
                elif (r, c) == start_pos:
                    color = COLOR_START
                elif (r, c) in goals:
                    color = COLOR_GOAL
                elif (r, c) in path_nodes:
                    color = COLOR_PATH 
                    
                # Rellenar mosaico
                x_start = c * tile_size
                y_start = r * tile_size
                for x in range(x_start, x_start + tile_size):
                    for y in range(y_start, y_start + tile_size):
                        pixels[x, y] = color
                        
        out_img.save(output_path)
        print(f"Path visualized on grid saved to {output_path}")
