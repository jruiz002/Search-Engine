from PIL import Image
import math
import numpy as np

# Definiendo constantes de color para facilitar la comparación
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

class Environment:
    """
    Maneja la discretización de la imagen en una cuadrícula de nodos.
    Implementación de la Tarea 1.1.
    """
    def __init__(self, image_path: str, tile_size: int = 20):
        self.image_path = image_path
        self.tile_size = tile_size
        self.image = Image.open(image_path).convert('RGB')
        self.width, self.height = self.image.size
        self.cols = math.ceil(self.width / tile_size)
        self.rows = math.ceil(self.height / tile_size)
        
        self.grid = np.zeros((self.rows, self.cols), dtype=int)
        # 0: Libre, 1: Obstáculo, 2: Inicio, 3: Meta
        
        self.start_pos = None
        self.goals = []
        
        self._process_image()

    def _process_image(self):
        """
        Itera sobre la imagen en lotes (mosaicos) e identifica el contenido de cada mosaico.
        """
        pixels = self.image.load()
        
        for r in range(self.rows):
            for c in range(self.cols):
                # Analizar el contenido del mosaico en (r, c)
                x_start = c * self.tile_size
                y_start = r * self.tile_size
                x_end = min(x_start + self.tile_size, self.width)
                y_end = min(y_start + self.tile_size, self.height)
                
                has_red = False
                has_green = False
                has_black = False
                
                count_black = 0
                total_pixels = (x_end - x_start) * (y_end - y_start)
                
                for x in range(x_start, x_end):
                    for y in range(y_start, y_end):
                        r_pix, g_pix, b_pix = pixels[x, y]
                        
                        if r_pix > 200 and g_pix < 50 and b_pix < 50:
                            has_red = True
                        
                        elif g_pix > 200 and r_pix < 50 and b_pix < 50:
                            has_green = True
                            
                        elif r_pix < 50 and g_pix < 50 and b_pix < 50:
                            has_black = True
                            count_black += 1
                
                
                if has_red:
                    self.grid[r, c] = 2
                    self.start_pos = (r, c)
                elif has_green:
                    self.grid[r, c] = 3
                    self.goals.append((r, c))
                elif has_black and (count_black / total_pixels) > 0.5:
                     self.grid[r, c] = 1
                else:
                    self.grid[r, c] = 0

    def get_start(self):
        """Devuelve la coordenada de inicio (fila, col)."""
        return self.start_pos

    def get_goals(self):
        """Devuelve la lista de coordenadas objetivo [(fila, col)]."""
        return self.goals

    def is_obstacle(self, row, col):
        """Devuelve True si el mosaico es un obstáculo."""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.grid[row, col] == 1
        return True # Fuera de límites es un obstáculo

    def get_grid_size(self):
        return self.rows, self.cols

    def show_discretized(self):
        """
        HERRAMIENTA DE DEPURACIÓN: Devuelve una nueva visualización de imagen de la cuadrícula discretizada.
        """
        out_img = Image.new('RGB', (self.cols * self.tile_size, self.rows * self.tile_size), 'white')
        pixels = out_img.load()
        
        for r in range(self.rows):
            for c in range(self.cols):
                color = WHITE
                val = self.grid[r, c]
                if val == 1: color = BLACK
                elif val == 2: color = RED
                elif val == 3: color = GREEN
                
                # Rellenar el mosaico
                x_start = c * self.tile_size
                y_start = r * self.tile_size
                for x in range(x_start, x_start + self.tile_size):
                    for y in range(y_start, y_start + self.tile_size):
                         pixels[x, y] = color
        return out_img
