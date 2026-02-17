import os
from environment import Environment
from problem import MazeProblem
from search import BFS, DFS, AStar
from visualization import Visualizer

def main():
    # Configuración
    assets_dir = "assets"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Obtener todos los archivos de imagen del directorio assets
    image_files = [f for f in os.listdir(assets_dir) if f.lower().endswith(('.png', '.bmp', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No image files found in {assets_dir}")
        return

    tile_size = 20 # Ajustar según sea necesario según la resolución de la imagen

    for image_file in image_files:
        input_image_path = os.path.join(assets_dir, image_file)
        # Crear un prefijo de nombre de archivo seguro (eliminar extensión)
        base_name = os.path.splitext(image_file)[0]
        
        # Crear subcarpeta para esta imagen específica
        image_output_dir = os.path.join(output_dir, base_name)
        os.makedirs(image_output_dir, exist_ok=True)
        
        print(f"\n{'='*40}")
        print(f"Processing {input_image_path} with tile size {tile_size}...")
        print(f"Results will be saved in {image_output_dir}")
        print(f"{'='*40}")
        
        # 1. Configuración del entorno y problema
        try:
            env = Environment(input_image_path, tile_size)
            problem = MazeProblem(env)
            visualizer = Visualizer(env)
        except Exception as e:
            print(f"Skipping {image_file}: {e}")
            continue
        
        # 2. BFS (Búsqueda en Anchura)
        print("\nRunning BFS...")
        bfs = BFS()
        solution_bfs = bfs.search(problem)
        if solution_bfs:
            print(f"BFS Solution Found! Cost: {solution_bfs.path_cost}")
            output_bfs = os.path.join(image_output_dir, "bfs_solution.png")
            visualizer.draw_path_on_original(solution_bfs, output_bfs)
        else:
            print("BFS: No solution found.")

        # 3. DFS (Búsqueda en Profundidad)
        print("\nRunning DFS...")
        dfs = DFS()
        solution_dfs = dfs.search(problem)
        if solution_dfs:
            print(f"DFS Solution Found! Cost: {solution_dfs.path_cost}")
            output_dfs = os.path.join(image_output_dir, "dfs_solution.png")
            visualizer.draw_path_on_original(solution_dfs, output_dfs)
        else:
            print("DFS: No solution found.")

        # 4. A* (A Estrella)
        print("\nRunning A*...")
        astar = AStar()
        solution_astar = astar.search(problem)
        if solution_astar:
            print(f"A* Solution Found! Cost: {solution_astar.path_cost}")
            output_astar = os.path.join(image_output_dir, "astar_solution.png")
            visualizer.draw_path_on_grid(solution_astar, output_astar)
        else:
            print("A*: No solution found.")

if __name__ == "__main__":
    main()
